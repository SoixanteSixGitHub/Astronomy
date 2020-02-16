import time
import copy
from utils import fix_map, macro_f1
from Model import *
from Dataset import *
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim import lr_scheduler


def test(load_path=None):
    print('testing', '---' * 20)
    test_path ='./data/val_sets_v1.csv'
    test_reader = pandas.read_csv(test_path, chunksize=4096)

    model = AstroNet()
    state_dict = torch.load(load_path)
    model.load_state_dict(state_dict)
    print("data and model is ready!!!")
    device = torch.device("cpu")
    model.eval()
    result = pandas.DataFrame({'id': [], 'label': []})
    label2cls = {0.0: 'star', 1.0: 'galaxy', 2.0: 'qso'}
    for bat_idx, test_df in enumerate(test_reader, 1):
        test_set = Astronomy(test_df, 'test')
        test_loader = DataLoader(test_set, batch_size=4096, shuffle=False, num_workers=8)
        for inputs, ids in test_loader:
            inputs.to(device)
        outputs = model(inputs)
        _idx, predictions = torch.max(outputs, dim=1)
        _result = pandas.DataFrame({'id': list(ids), 'label_': list(predictions.float())})
        _result['label'] = _result['label_'].map(label2cls)
        result = pandas.concat([result, _result], axis=0)
    result.to_csv('./results/test.csv', index=False, encoding='utf-8')
    fix_map('./results/test.csv')  # fixing some map problem....


def train_val(load_path = None):
    '''main funtion to train and validate '''
    path = './data/'
    train_path, test_path = path + 'new_columns_trains_sets.csv', path + 'val_sets_v1.csv'
    model = AstroNet()
    print(model)
    if load_path:
        state_dict = torch.load(load_path)  # print(state_dict)
        model.load_state_dict(state_dict)
    print("data and model is ready!!!")

    ''' loss_function ------ optimizer ------->  train '''
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)    # 0.0001?
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[6,8], gamma=0.1)  # to adjust LR
    epoches = 12
    device = torch.device("cpu")  # torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)      # # of course we have Gpu!! but not necessary!
    model_dir = "./checkpoints/"
    # training---------------------------------------------------------------------------------------------------------
    print("ready to train!!", "Current Time : " + time.ctime())
    best_model_wts = copy.deepcopy(model.state_dict())
    best_f1 = 0.0
    for epoch in range(1, epoches+1):
        print('Epoch {}/{}'.format(epoch, epoches),'--' * 10)
        running_loss,running_corrects, data_num = 0.0, 0.0, 0.0
        train_reader = pandas.read_csv(train_path, chunksize=256, low_memory=False)
        '''one thing to be clarified here is that since the csv file is too large to load, we have to read it chunk by chunk, 
            each chunk (whose type is pandas.DataFrame) is used to initialize a Astronomy Dataset defined in nn_ModelSet.py'''
        scheduler.step()  # adjust this to end of each epoch if using pytorch>1.1...
        model.train()     # Set model to training mode
        for batch_idx, train_df in enumerate(train_reader, 1):
            try:
                train_set = Astronomy(train_df)  # each chunk is used to initialize a Astronomy Dataset
                train_loader = DataLoader(train_set, batch_size=256, shuffle=True, num_workers=8)
                # actually one batch though 'for' is used:
                for inputs, labels, ids in train_loader:
                    inputs = inputs.to(device)
                    labels = labels.to(device)
            except ValueError:  # there seems to be that some training data has data_type problem, i choose to drop them >~<
                continue
            # zero the parameter gradients
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print something like loss and precision is also important!!
            _value, predictions = torch.max(outputs, dim=1)  # common trick here!!!
            batch_num, batch_corrects = len(outputs), torch.sum(predictions == labels.data)
            print('->batch NO.#%d\tbatch_size:%d\tavg_loss:%.4f\tacc:%.4f ' % (batch_idx, batch_num, loss.item(), batch_corrects.double()/batch_num),time.ctime())
            running_corrects, running_loss, data_num = running_corrects+batch_corrects.item(), running_loss+loss.item()*batch_num, data_num+batch_num

        epoch_loss, epoch_acc = running_loss/data_num, running_corrects/data_num  # averaged!
        print('Epoch {}/{}'.format(epoch, epoches),'Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc), 'data_num: ', data_num)

        # evaluating------------------------------------------------------------------------------------------------------------
        print('evaluating', '---'*20)
        model.eval()
        test_reader = pandas.read_csv(test_path, chunksize=4096)
        result = pandas.DataFrame({'id':[], 'label':[]})
        label2cls = {0.0:'star', 1.0:'galaxy', 2.0:'qso'}
        for bat_idx, test_df in enumerate(test_reader, 1):
            test_set = Astronomy(test_df, 'test')  # each chunk is used to initialize a Astronomy Dataset
            test_loader = DataLoader(test_set, batch_size=4096, shuffle=False, num_workers=8)
            for inputs, ids in test_loader:
                inputs.to(device)
            outputs = model(inputs)
            _idx, predictions = torch.max(outputs, dim=1)
            _result = pandas.DataFrame({'id':list(ids), 'label_':list(predictions.float())})
            _result['label'] = _result['label_'].map(label2cls)  # some map problem happens, i dont know why but i will fix it later
            result = pandas.concat([result, _result], axis=0)
        result.to_csv('./results/Feb16-evaluate_epoch%d.csv' % epoch, index=False, encoding='utf-8')
        fix_map('./results/evaluate_epoch%d.csv' % epoch)  # fixing some map problem...
        # eval and deep copy the model
        print('Answer written to disk & Start Macro-F1 evaluating!')
        epoch_f1 = macro_f1('./results/evaluate_epoch%d.csv' % epoch, './logs/Feb16-evaluate_epoch%d.json' % epoch)
        if epoch_f1 > 0: # i save model each epoch, chang this line to 'if epoch_f1 > best_f1:' if you just want best model on validate_set
                best_f1 = epoch_f1
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(best_model_wts, model_dir + "Feb16-epoch%d-macroF1-%.4f.pth" % (epoch, best_f1))

    print('Best macro f1 : {:4f}'.format(best_f1), "Current Time : " + time.ctime())


if __name__ == '__main__':
    train_val()

