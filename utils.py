import pandas
import json
def macro_f1(answer_pth='', log_path=None):
    ''' this is a utility function to evaluate the answer (a csv file to submit)
        according to macro F1 '''
    path = './results/'
    std_answer = pandas.read_csv(path+'val_labels_v1.csv')
    my_answer = pandas.read_csv(answer_pth)
    print('Answers loaded！！')
    # TP my std
    std = {'star':0.0, 'galaxy':0.0, 'qso':0.0}
    my  = {'star':0.0, 'galaxy':0.0, 'qso':0.0}
    TP  = {'star':0.0, 'galaxy':0.0, 'qso':0.0}
    precision = {'star':0.0, 'galaxy':0.0, 'qso':0.0}
    recall    = {'star':0.0, 'galaxy':0.0, 'qso':0.0}
    F1        = {'star':0.0, 'galaxy':0.0, 'qso':0.0}

    for i in range(len(std_answer)):
        if std_answer.iloc[i]['id'] != std_answer.iloc[i]['id']:
            print('Not same ID order!')
            continue
        if not (my_answer.iloc[i]['label'] in ('star','galaxy','qso')):
            print('my Not correct answer!')
            continue
        # real game begins!
        std[std_answer.iloc[i]['label']] += 1
        my[my_answer.iloc[i]['label']] += 1
        if std_answer.iloc[i]['label'] == my_answer.iloc[i]['label']:
            TP[std_answer.iloc[i]['label']] +=1
    print('std_answer: ', std)
    print('my_answer: ', my)
    print('TP_answer: ', TP)
    # precision and recall
    for label in ('star','galaxy','qso'):
        # to avoid zero
        if my[label] == 0:
            my[label] = 1
        precision[label] = TP[label] / my[label]
        recall[label] = TP[label] / std[label]
    print('precision: ', precision)
    print('recall: ', recall)
    # then F1
    for label in ('star','galaxy','qso'):
        if precision[label]==0 or recall[label]==0:
            F1[label] = 0
        else:
            F1[label] = 2 * precision[label] * recall[label]/ (recall[label] + precision[label])
    print('F1: ', F1)
    print('std_valid: ', sum(std.values()), 'my_valid: ', sum(my.values()))

    if log_path:
        eval_log = {
            'std': std,
            'my': my,
            'TP': TP,
            'precision': precision,
            'recall': recall,
            'F1': F1,
        'macro-F1':sum(F1.values())/3}
        with open(log_path, 'w') as f:
            json.dump(eval_log, f)

    return sum(F1.values())/3

def fix_map(answer_path):
    label2cls = {'tensor(0.)':'star', 'tensor(1.)':'galaxy', 'tensor(2.)':'qso'}
    answer = pandas.read_csv(answer_path)
    for i in range(len(answer)):
        if not (answer.iloc[i]['label'] in ('star', 'galaxy', 'qso')):
            answer.iloc[i]['label'] = label2cls[answer.iloc[i]['label_']]
    answer = answer.drop(['label_'], axis=1)
    answer.to_csv(answer_path, index=False, encoding='utf-8')


if __name__ == '__main__':
    # fix_map('./results/sample_submission.csv')
    score = macro_f1('./results/val_labels_v1.csv')  # evaluate_epoch1   sample_submission
    print(score)


