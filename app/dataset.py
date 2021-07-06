# import libraries
import torch, torchvision
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
# from torch.utils.data.dataset import Dataset
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.nn as nn
import numpy as np
import csv
import inspect
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, precision_recall_fscore_support


class ImageDataset(Dataset):
    '''
    A Dataset class for images
    inputs : numpy array [n_data x shape]
    labels : numpy array [n_data (x 1)]
    '''
    def __init__(self, dataset):
        self.dataset = dataset
        self.test_dataset = None

    def __getitem__(self, index):
        train_loader, test_loader = self.load_data(self.dataset)
        # train_loader, test_loader = self.load_data_lstm(self.dataset)

        return (train_loader, test_loader)


    def pad_sequence(self, sequence, pad_value=-10.0, timesteps=10, dimensions=4):
        N = len(sequence)
        X = sequence

        padded = np.full((N, timesteps, dimensions), fill_value=pad_value)
        
        for s, x in enumerate(X):
            seq_len = x.shape[0]
            padded[s, 0:seq_len, :] = x
        return padded

    
    def load_data(self):
        BATCH_SIZE_TRAIN = 100
        BATCH_SIZE_TEST = 64

        dataset = self.dataset
        train_loader = None
        test_loader = None

        print("loading dataset ....")

        '''
        Return MNIST train/test data and labels as numpy arrays
        '''
        if dataset == 'MNIST':
            # Loading MNIST using torchvision.datasets
            train_loader = torch.utils.data.DataLoader(
                torchvision.datasets.MNIST('./data', train=True, download=True,
                                            transform=torchvision.transforms.Compose([
                                            torchvision.transforms.ToTensor(),
                                            torchvision.transforms.Normalize(
                                                (0.1307,), (0.3081,))
                                            ])),
                batch_size=BATCH_SIZE_TRAIN, shuffle=True)

            test_loader = torch.utils.data.DataLoader(
                torchvision.datasets.MNIST('./data', train=False, download=True,
                                            transform=torchvision.transforms.Compose([
                                            torchvision.transforms.ToTensor(),
                                            torchvision.transforms.Normalize(
                                                (0.1307,), (0.3081,))
                                            ])),
                batch_size=BATCH_SIZE_TEST, shuffle=False)

        elif dataset == 'CIFAR':
            train_loader = torch.utils.data.DataLoader(
                torchvision.datasets.CIFAR10('./data', train=True, download=True,
                                            transform=torchvision.transforms.Compose([
                                            transforms.RandomCrop(32, padding=4),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                            ])),
                batch_size=BATCH_SIZE_TRAIN, shuffle=True)

            # Normalize the test set same as training set without augmentation
            test_loader = torch.utils.data.DataLoader(
                torchvision.datasets.CIFAR10('./data', train=False, download=True,
                                            transform=torchvision.transforms.Compose([
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                            ])),
                batch_size=BATCH_SIZE_TEST, shuffle=False)

        return train_loader, test_loader


    def load_data_lstm(self, label_file='data/labels.npy'):

        with open(self.dataset, 'rb') as f:
            file_x = np.load(f, allow_pickle=True)

        with open(label_file, 'rb') as f:
            file_y = np.load(f, allow_pickle=True)
    
        train_x = file_x
        train_y = file_y

        # Split the Train-Test
        X_trainval, X_test, y_trainval, y_test = train_test_split(train_x, train_y, test_size=0.2)
        # Split train into train-val
        X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=0.1, random_state=21)

        train_dataset = CustomDataset(X_trainval, y_trainval)
        test_dataset = CustomDataset(X_test, y_test)
        val_dataset = CustomDataset(X_val, y_val)

        self.test_dataset = X_test, y_test

        BATCH_SIZE = 16
        train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(dataset=test_dataset, batch_size=1)
        val_loader = DataLoader(dataset=val_dataset, batch_size=1)

        return train_loader, test_loader, val_loader


    def test_lstm_model (self, model, test_loader):           

        # model.eval()

        # with torch.no_grad():
            
        #     for data, target in enumerate(self.test_dataset):
        #         data = torch.Tensor(data, dtype=torch.float32)
        #         y_test_pred = model( data )
        #         print( y_test_pred )
        pass


        # x_test, y_test = self.test_dataset
        
        # acc = self.accuracy(model, x_test, y_test, 0.15)

        # return acc

        # model.eval()
        # with torch.no_grad():
        #     for i in range(0, len(x_test)):
        #         y_test_pred = model(x_test[i])
        #         print(y_test_pred)
        # with torch.no_grad():
        #     model.eval()

        #     for batch_idx, (data, target) in enumerate(test_loader.dataset):
        #         y_test_pred = model(X_test)
        #         print(y_test_pred)

        # for i, (data, target) in enumerate(self.test_dataset):
        #     x_batch = torch.tensor(data)
        #     print(x_batch)
            # y_test_pred = model(data)
            # x_batch = torch.tensor(data, dtype=torch.float32)
            # y_batch = torch.tensor(target, dtype=torch.float32)

        # y_pred_list = []
        # y_test = []
        
        # for data, target in self.test_dataset:
        #     y_test_pred = model(data)
        #     y_pred_list.append(y_test_pred.numpy())
        #     # print(data, target)
        #     # x_batch = torch.tensor(data, dtype=torch.float32)
        #     # y_test_pred = model(x_batch)
        #     print(y_test_pred)

        # TESTING
        # y_pred_list = []
        # y_test = []
        # with torch.no_grad():
        #     mfor batch_idx, (data, target) in enumerate(test_loader):odel.eval()
        #     
        #         x_batch = torch.tensor(data, dtype=torch.float32)
        #         y_test_pred = model(x_batch)
        #         y_pred_list.append(y_test_pred.cpu().numpy())
        #         y_test.append(target)
        # y_pred_list = [a.squeeze().tolist() for a in y_pred_list] 
        # y_test = [a.squeeze().tolist() for a in y_test] 
        # test_mse = mean_squared_error(y_test, y_pred_list)
        # print("Mean Squared Error :",test_mse)
        # pass

        # testing_dataset = torch.utils.data.TensorDataset(X_trainval, y_trainval)
        # test_loader = torch.utils.data.DataLoader(testing_dataset, batch_size=1)
        
        # for i in range(0, len(X_test)):
        #     X_batch = X_test[i]
        #     output = model(X_batch)
            # print(output)
        # y_pred_list = []
        # with torch.no_grad():
        #     model.eval()

        # for batch_idx, (data, target) in enumerate(test_loader):
        #     print(data)
        #         X_batch = X_batch.to(device)
        #         y_test_pred = model(X_batch)
        #         y_pred_list.append(y_test_pred.cpu().numpy())
        # y_pred_list = [a.squeeze().tolist() for a in y_pred_list] 
        # test_mse = mean_squared_error(y_test, y_pred_list)



    def test_model(self, model, test_loader):
        model.eval()
        test_loss = 0
        correct = 0
        test_losses = []

        with torch.no_grad():
            for data, target in test_loader:
                output = model(data)
                test_loss += F.nll_loss(output, target, size_average=False).item()
                pred = output.data.max(1, keepdim=True)[1]  # get class from network's prediction
                correct += pred.eq(target.data.view_as(pred)).sum()
        
        test_loss /= len(test_loader.dataset)
        test_losses.append(test_loss)

        # accuracy = 100. * correct / len(test_loader.dataset)
        accuracy = correct / len(test_loader.dataset)
        accuracy = float(accuracy)

        return accuracy

    
    def test_model_detail (self, model, test_loader):
        model.eval()
        test_loss = 0
        correct = 0
        test_losses = []
        target_true = 0
        predicted_true = 0
        correct_true = 0
        precision = 0
        recall = 0

        with torch.no_grad():
            for data, target in test_loader:
                output = model(data)
                test_loss += F.nll_loss(output, target, size_average=False).item()  # sum up batch loss
                pred = output.data.max(1, keepdim=True)[1]   # get the index of the max log-probability
                correct += pred.eq(target.data.view_as(pred)).sum()

                precision+=precision_score(target.data.view_as(pred).cpu(),pred.cpu(), average='macro')
                recall+=recall_score(target.data.view_as(pred).cpu(),pred.cpu(), average='macro')

                # modifications for precision and recall
                predicted_classes = torch.argmax(output, dim=1) == 0
                
        
        test_loss /= len(test_loader.dataset)
        test_losses.append(test_loss)

        precision /= len(test_loader.dataset)

        accuracy = correct / len(test_loader.dataset)
        accuracy = float(accuracy)

        return accuracy, precision, recall



    def test_precision(self, model, test_loader):

        pred = []
        true = []
        sm = nn.Softmax(dim = 1)
        with torch.no_grad():
            model.eval()

            for batch_idx, (data, target) in enumerate(test_loader):
                output = model(data)
                output = sm(output)
                _, preds = torch.max(output, 1)
                preds = preds.cpu().numpy()
                target = target.cpu().numpy()
                preds = np.reshape(preds,(len(preds),1))
                target = np.reshape(target,(len(preds),1))

                for i in range(len(preds)):
                    pred.append(preds[i])
                    true.append(target[i])

        precision = precision_score(true,pred,average='macro')
        recall = recall_score(true,pred,average='macro')
        accuracy = accuracy_score(true,pred)
        return accuracy, precision, recall

    
    def test_validations(self, model, test_loader):

        # ----------------- VALIDATION  ----------------- 
        val_losses = 0
        precision, recall = 0, 0
        
        # set model to evaluating (testing)
        y_pred_list = []
        target_list = []
        model.eval()
        with torch.no_grad():
            for data, target in test_loader:
                y_test_pred = model(data)
                y_pred_tag = torch.round(y_test_pred)
                y_pred_list.append(y_pred_tag.cpu().numpy())
                target_list.append(target)

        y_pred_list = [a.squeeze().tolist() for a in y_pred_list]
        precision = precision_score(target_list, y_pred_list, average='macro')
        recall = recall_score(target_list, y_pred_list, average='macro')

        return precision, recall


    def train_model(self, model, train_loader, learning_rate = 0.01, momentum = 0.5, epochs = 1, iteration=1):
        '''
        Update the client model on client data
        '''
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum) # define optimizer

        # unfreeze layers
        for param in model.parameters():
            param.requires_grad = True


        # train model
        model.train()
        for epoch in range(epochs):
            for batch_idx, (data, target) in enumerate(train_loader):
                optimizer.zero_grad()
                output = model(data)
                loss = F.nll_loss(output, target)
                loss.backward()
                optimizer.step()
                print('Iteration: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        iteration, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))
            
            print("Training completed for local model for iteration {}, epoch {}".format(iteration, epoch))


    def train_lstm_model(self, model, train_loader, test_loader, learning_rate = 0.01, momentum = 0.5, epochs = 3, iteration=1):
        '''
        train the LSTM model
        '''
        num_epochs = 200
        batch_size = 16
        criterion = torch.nn.MSELoss(reduction='mean')
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        mse_list = []

        loss_stats = {
            'train': [],
            "val": [],
            'test': []
        }

        # train the model
        for e in range(epochs):
            train_epoch_loss = 0

            for batch_idx, (data, target) in enumerate(train_loader):

                x_batch = torch.tensor(data, dtype=torch.float32)
                y_batch = torch.tensor(target, dtype=torch.float32)

                optimizer.zero_grad()
                output = model(x_batch)
                train_loss = criterion(output, y_batch)
                
                train_loss.backward()
                optimizer.step()
                
                train_epoch_loss += train_loss.item()
            print('step : ' , e , 'loss : ' , train_loss.item())


            # TESTING
            y_pred_list = []
            y_test = []
            with torch.no_grad():
                model.eval()
                for batch_idx, (data, target) in enumerate(test_loader):
                    x_batch = torch.tensor(data, dtype=torch.float32)
                    y_test_pred = model(x_batch)
                    y_pred_list.append(y_test_pred.cpu().numpy())
                    y_test.append(target)
            y_pred_list = [a.squeeze().tolist() for a in y_pred_list] 
            y_test = [a.squeeze().tolist() for a in y_test] 
            test_mse = mean_squared_error(y_test, y_pred_list)
            mse_list.append(test_mse)
            print("Mean Squared Error :",test_mse)

        return test_mse
            
            # print(f'Epoch {e+0:03}: | Train Loss: {train_epoch_loss/len(train_loader):.5f} | Val Loss: {val_epoch_loss/len(val_loader):.5f}')


    def train_lstm_attack(self, model, train_loader, test_loader, learning_rate = 0.01, momentum = 0.5, epochs = 3, iteration=1):
        '''
        train the LSTM model
        '''
        num_epochs = 200
        batch_size = 16
        criterion = torch.nn.MSELoss(reduction='mean')
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        mse_list = []

        loss_stats = {
            'train': [],
            "val": [],
            'test': []
        }

        # freeze layers
        for param in model.parameters():
            param.requires_grad = False

        # train the model
        for e in range(epochs):
            train_epoch_loss = 0

            for batch_idx, (data, target) in enumerate(train_loader):

                x_batch = torch.tensor(data, dtype=torch.float32)
                y_batch = torch.tensor(target, dtype=torch.float32)

                optimizer.zero_grad()
                output = model(x_batch)
                train_loss = criterion(output, y_batch)
                
                # train_loss.backward()
                # optimizer.step()
                
                train_epoch_loss += train_loss.item()
            print('step : ' , e , 'loss : ' , train_loss.item())


            # TESTING
            y_pred_list = []
            y_test = []
            with torch.no_grad():
                model.eval()
                for batch_idx, (data, target) in enumerate(test_loader):
                    x_batch = torch.tensor(data, dtype=torch.float32)
                    y_test_pred = model(x_batch)
                    y_pred_list.append(y_test_pred.cpu().numpy())
                    y_test.append(target)
            y_pred_list = [a.squeeze().tolist() for a in y_pred_list] 
            y_test = [a.squeeze().tolist() for a in y_test] 
            test_mse = mean_squared_error(y_test, y_pred_list)
            mse_list.append(test_mse)
            print("Mean Squared Error :",test_mse)

        return test_mse


    def train_attack(self, classifer, train_loader, learning_rate = 0.01, momentum = 0.5, epochs = 1, iteration=1):
        '''
        Update the client model on client data
        '''
        optimizer = optim.SGD(classifer.parameters(), lr=learning_rate, momentum=momentum) # define optimizer
        
        # freeze layers
        for param in classifer.parameters():
            param.requires_grad = False

        # train model
        classifer.train()
        for epoch in range(epochs):
            for batch_idx, (data, target) in enumerate(train_loader):
                optimizer.zero_grad()
                output = classifer(data)
                loss = F.nll_loss(output, target)
                # loss.backward()
                # optimizer.step()
                print('Iteration: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        iteration, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))
            
            print("Training completed for local model for iteration {}, epoch {}".format(iteration, epoch))
        


    def accuracy(self, model, data_x, data_y, pct_close):

        n_items = len(data_y)
        X = torch.Tensor(data_x)
        Y = torch.Tensor(data_y)
        output = model(X)
        pred = output.view(n_items)
        n_correct = torch.sum((torch.abs(pred - Y) < torch.abs(pct_close * Y)))
        acc = (n_correct.item() * 100.0 / n_items)
        
        return acc

    
    def calculate_metric(self, metric_fn, true_y, pred_y):
        if "average" in inspect.getfullargspec(metric_fn).args:
            return metric_fn(true_y, pred_y, average="macro")
        else:
            return metric_fn(true_y, pred_y, average="macro")
        

    def print_scores(self, p, r, f1, a, batch_size):
        for name, scores in zip(("precision", "recall", "F1", "accuracy"), (p, r, f1, a)):
            print(f"\t{name.rjust(14, ' ')}: {sum(scores)/batch_size:.4f}")



class CustomDataset(Dataset):
    
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data

    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]
        
    def __len__ (self):
        return len(self.X_data)    

    

class CSVWriter():

    filename = None
    fp = None
    writer = None

    def __init__(self, filename):
        self.filename = filename
        self.fp = open(self.filename, 'w', encoding='utf8')
        self.writer = csv.writer(self.fp, delimiter=';', quotechar='"', quoting=csv.QUOTE_ALL, lineterminator='\n')

    def close(self):
        self.fp.close()

    def write(self, elems):
        self.writer.writerow(elems)

    def size(self):
        return os.path.getsize(self.filename)

    def fname(self):
        return self.filename