import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from models import CNN

class Trainer:
    def __init__(self, source_train_loader, source_test_loader, target_train_loader, target_test_loader, device='cuda'):
        self.model = CNN()
        if torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)  # Multi-GPU support
        self.model = self.model.to(device)

        self.discriminator = None
        self.source_train_loader = source_train_loader
        self.source_test_loader = source_test_loader
        self.target_train_loader = target_train_loader
        self.target_test_loader = target_test_loader
        self.device = device
        self.source_prototypes = {}
        self.target_prototypes = {}
        self.new_source_prototypes = {}
        self.new_target_prototypes = {}
        
        self.criterion_clf = nn.CrossEntropyLoss()
        self.criterion_fa = nn.MSELoss()
        
        # self.pt = True
        self.pt = False

    def _init_prototypes_target(self, similarity_threshold=0.7):
        self.model.eval()

        with torch.no_grad():
            for data in self.target_train_loader:
                inputs, _ = data
                inputs = inputs.to(self.device)

                target_features = self.model(inputs)['features']

                for feature in target_features:
                    similarities = []
                    
                    for class_label, prototype in self.source_prototypes.items():
                        similarity = F.cosine_similarity(feature.unsqueeze(0), prototype.unsqueeze(0))
                        similarities.append((class_label, similarity.item()))
                    
                    similarities.sort(key=lambda x: x[1], reverse=True)
                    pseudo_label = similarities[0][0]

                    if pseudo_label not in self.target_prototypes:
                        self.target_prototypes[pseudo_label] = []
                    
                    if similarities[0][1] >= similarity_threshold:
                        self.target_prototypes[pseudo_label].append(feature)

        for class_label in self.target_prototypes:
            features = self.target_prototypes[class_label]
            if features:
                avg_prototype = torch.stack(features).mean(dim=0)
                self.target_prototypes[class_label] = avg_prototype
                print("Target prototype for class", class_label, "initialized.")
        
        # Clear the target_train_loader to prevent data leakage
        torch.cuda.empty_cache()

    def _init_prototypes_source(self):
        self.model.eval()

        class_feature_sum = {}
        class_instance_count = {}

        with torch.no_grad():
            for data in self.source_train_loader:
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                features = self.model(inputs)['features']

                for feature, label in zip(features, labels):
                    label = label.item()
                    if label not in class_feature_sum:
                        class_feature_sum[label] = torch.zeros_like(feature).to(self.device)
                        class_instance_count[label] = 0

                    class_feature_sum[label] += feature
                    class_instance_count[label] += 1

        for label in class_feature_sum:
            self.source_prototypes[label] = class_feature_sum[label] / class_instance_count[label]
            print("Prototypes for", label, "source domain created.")
        
        # Clear the source_train_loader to prevent data leakage
        torch.cuda.empty_cache()

    def _pre_train(self, epochs=15):
        self.model.train()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)  

        for epoch in range(epochs):
            for data in self.source_train_loader:
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                optimizer.zero_grad()

                outputs = self.model(inputs)['logits']
                loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()

            print('Epoch', epoch + 1, '/', epochs)            
            self.test()
            torch.save(self.model.state_dict(), 'pretrained_model.pth')
        
        # Clear the source_train_loader to prevent data leakage
        torch.cuda.empty_cache()

    def _update_prototype(self):
        def update_prototypes(original_prototypes, new_prototypes):
            for class_label, new_prototype in new_prototypes.items():
                if class_label in original_prototypes:
                    original_prototype = original_prototypes[class_label]
                    cosine_dist = F.cosine_similarity(original_prototype.unsqueeze(0), new_prototype.unsqueeze(0)).item()
                    weight_new = cosine_dist ** 2
                    weight_old = 1 - weight_new
                    updated_prototype = weight_old * original_prototype + weight_new * new_prototype
                    original_prototypes[class_label] = updated_prototype
                else:
                    original_prototypes[class_label] = new_prototype

        update_prototypes(self.source_prototypes, self.new_source_prototypes)
        update_prototypes(self.target_prototypes, self.new_target_prototypes)

        self.new_source_prototypes.clear()
        self.new_target_prototypes.clear()
        torch.cuda.empty_cache()

    def _calculate_new_prototypes(self, source_features, target_features, source_labels, threshold=0.7):
        self.new_source_prototypes = {}
        self.new_target_prototypes = {}

        for feature, label in zip(source_features, source_labels):
            label = label.item()
            if label not in self.new_source_prototypes:
                self.new_source_prototypes[label] = []
            self.new_source_prototypes[label].append(feature)

        for feature in target_features:
            similarities = []
            for class_label, prototype in self.source_prototypes.items():
                similarity = F.cosine_similarity(feature.unsqueeze(0), prototype.unsqueeze(0))
                similarities.append((class_label, similarity.item()))
            similarities.sort(key=lambda x: x[1], reverse=True)
            pseudo_label = similarities[0][0]
            if pseudo_label not in self.new_target_prototypes:
                self.new_target_prototypes[pseudo_label] = []
            if similarities[0][1] >= threshold:
                self.new_target_prototypes[pseudo_label].append(feature)

        self.new_source_prototypes = {label: torch.stack(features).mean(dim=0) for label, features in self.new_source_prototypes.items()}
        self.new_target_prototypes = {label: torch.stack(features).mean(dim=0) for label, features in self.new_target_prototypes.items() if features}

    def _train_epoch(self, epoch, optimizer_m):
        self.model.train()
        running_loss_clf = 0.0
        running_loss_fa = 0.0

        for i, (source_data, target_data) in enumerate(zip(self.source_train_loader, self.target_train_loader)):
            source_inputs, source_labels = source_data
            target_inputs, _ = target_data
            source_inputs, source_labels = source_inputs.to(self.device), source_labels.to(self.device)
            target_inputs = target_inputs.to(self.device)

            source_outputs = self.model(source_inputs)
            target_outputs = self.model(target_inputs)

            source_features = source_outputs['features']
            target_features = target_outputs['features']

            self._calculate_new_prototypes(source_features, target_features, source_labels)
            self._update_prototype()

            loss_clf = self.criterion_clf(source_outputs['logits'], source_labels)
            loss_fa = self.criterion_fa(torch.stack([p.detach() for p in self.source_prototypes.values()]), torch.stack([p.detach() for p in self.target_prototypes.values()]))
            loss = (loss_clf + loss_fa)

            loss.backward()

            optimizer_m.step()
            optimizer_m.zero_grad()

            running_loss_clf += loss_clf.item()
            running_loss_fa += loss_fa.item()

            del source_inputs, target_inputs, source_outputs, target_outputs, source_features, target_features, self.new_source_prototypes, self.new_target_prototypes
            torch.cuda.empty_cache()
            
        print('Epoch', epoch + 1, '/', self.epochs, 'Loss Clf:', round(running_loss_clf, 4), 'Loss FA:', round(running_loss_fa, 4))

    def train(self, epochs=15):
        self.epochs = epochs
        if self.pt:
            self._pre_train()
        else:
            self.model.load_state_dict(torch.load('/csehome/karki.1/BTP/feat_align_DA/pretrained_model.pth'))
        self._init_prototypes_source()
        self._init_prototypes_target()
        
        optimizer_m = optim.Adam(self.model.parameters(), lr=0.001)
        
        for epoch in range(epochs):
            self._train_epoch(epoch, optimizer_m)
            
            # self.model.to('cpu')
            self.test()
            # self.model.to(self.device)
            
            torch.cuda.empty_cache()

    def test(self):
        self.model.eval()
        criterion = nn.CrossEntropyLoss()

        loaders = {
            'Source Train': self.source_train_loader,
            'Source Test': self.source_test_loader,
            'Target Train': self.target_train_loader,
            'Target Test': self.target_test_loader
        }

        for name, loader in loaders.items():
            correct = 0
            total = 0
            running_loss = 0.0
            
            with torch.no_grad():
                for data in loader:
                    inputs, labels = data
                    inputs, labels = inputs.to(self.device), labels.to(self.device)

                    outputs = self.model(inputs)['logits']
                    loss = criterion(outputs, labels)

                    running_loss += loss.item()
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            accuracy = 100 * correct / total
            print(name, 'Accuracy:', round(accuracy, 2), '%, Loss:', round(running_loss, 4))

            del inputs, labels, outputs
            torch.cuda.empty_cache()
