import torch
import numpy as np
from torch.autograd import Variable
import parameters as par

# '<=50K' is 0 (majority); '>50K' is 1 (minority)


def accuracy_calculate(data_loader, model):
    correct = 0
    total = 0
    for i, (data, label) in enumerate(data_loader):
        data = Variable(data)
        label = Variable(label).type(torch.float32)
        output = model(data)
        if output.detach().numpy()[0] >= 0.5:
            pred = 1
            total += 1
            if pred == label.detach().numpy()[0]:
                correct += 1

        else:
            pred = 0
            total += 1
            if pred == label.detach().numpy()[0]:
                correct += 1
    acc = correct / total
    print("correct/total:(%s/%s)" % (correct, total))
    print("accuracy=", acc)


def balanced_accuracy(dataloader, model):
    correct = 0
    total = 0
    tp = 0
    tn = 0
    P = 0
    N = 0
    for i, (data, label) in enumerate(dataloader):
        data = Variable(data)
        label = Variable(label).type(torch.float32)
        output = model(data)
        if output.detach.numpy()[0] >= 0.5:
            pred = 1
            total += 1
            P += 1
            if pred == label.detach().numpy()[0]:
                correct += 1
                tp += 1
        else:
            pred = 0
            total += 1
            N += 1
            if pred == label.detach().numpy()[0]:
                correct += 1
                tn += 1
    acc = (tp + tn)/(P + N)
    print("correct/total:(%s/%s)" % (correct, total))
    print("tp + tn:", (tp + tn))
    print("p + n:", (P + N))
    print("acc:", acc)


def accuracy_calculate_gender(dataset, model):
    # single
    single_correct = 0
    married_correct = 0
    single_total = 0
    married_total = 0
    married = torch.tensor([1, 0]).type(torch.float32)
    single = torch.tensor([0, 1]).type(torch.float32)

    for i, (data, label) in enumerate(dataset):
        data = Variable(data)
        gt = Variable(label)
        label = torch.tensor([gt[0]])
        marital = gt[1:3]
        output = model(data)
        if torch.equal(marital, married) is True:
            married_total += 1
            if output.detach().numpy()[0] >= par.SIG_THRESHOLD:
                pred = 1
                if pred == label.detach().numpy()[0]:
                    married_correct += 1
            else:
                pred = 0
                if pred == label.detach().numpy()[0]:
                    married_correct += 1
        elif torch.equal(marital, single) is True:
            single_total += 1
            if output.detach().numpy()[0] >= par.SIG_THRESHOLD:
                pred = 1
                if pred == label.detach().numpy()[0]:
                    single_correct += 1
            else:
                pred = 0
                if pred == label.detach().numpy()[0]:
                    single_correct += 1

    single_acc = single_correct / single_total
    married_acc = married_correct / married_total
    total_acc = (single_correct+married_correct)/(single_total + married_total)
    # print("female income predict accuracy:%s" % female_acc)
    # print("male income predict accuracy:%s" % male_acc)
    # print("total predict accuracy:%s" % ((male_correct+female_correct)/total))
    return total_acc, single_acc, married_acc


def accuracy_calculate_gender_ae(dataset, model):
    # single
    single_correct = 0
    married_correct = 0
    single_total = 0
    married_total = 0
    married = torch.tensor([1, 0]).type(torch.float32)
    single = torch.tensor([0, 1]).type(torch.float32)

    for i, (data, label) in enumerate(dataset):
        data = Variable(data)
        gt = Variable(label)
        label = torch.tensor([gt[0]])
        marital = gt[1:3]
        _, _, output = model(data)
        if torch.equal(marital, married) is True:
            married_total += 1
            if output.detach().numpy()[0] >= par.SIG_THRESHOLD:
                pred = 1
                if pred == label.detach().numpy()[0]:
                    married_correct += 1
            else:
                pred = 0
                if pred == label.detach().numpy()[0]:
                    married_correct += 1
        elif torch.equal(marital, single) is True:
            single_total += 1
            if output.detach().numpy()[0] >= par.SIG_THRESHOLD:
                pred = 1
                if pred == label.detach().numpy()[0]:
                    single_correct += 1
            else:
                pred = 0
                if pred == label.detach().numpy()[0]:
                    single_correct += 1

    single_acc = single_correct / single_total
    married_acc = married_correct / married_total
    total_acc = (single_correct+married_correct)/(single_total + married_total)
    # print("female income predict accuracy:%s" % female_acc)
    # print("male income predict accuracy:%s" % male_acc)
    # print("total predict accuracy:%s" % ((male_correct+female_correct)/total))
    return total_acc, single_acc, married_acc




def balanced_accuracy_marital(dataset, model):
    single_tp, single_tn = 0, 0
    married_tp, married_tn = 0, 0
    single_p, single_n = 0, 0
    married_p, married_n = 0, 0
    married = torch.tensor([1, 0]).type(torch.float32)
    single = torch.tensor([0, 1]).type(torch.float32)

    for i, (data, label) in enumerate(dataset):
        data = Variable(data)
        gt = Variable(label)
        label = torch.tensor([gt[0]])
        marital = gt[1:3]
        output = model(data)
        if torch.equal(marital, married) is True:
            if output.detach().numpy()[0] >= par.SIG_THRESHOLD:
                pred = 1
                married_p += 1
                if pred == label.detach().numpy()[0]:
                    married_tp += 1
            else:
                pred = 0
                married_n += 1
                if pred == label.detach().numpy()[0]:
                    married_tn += 1
        elif torch.equal(marital, single) is True:
            if output.detach().numpy()[0] >= par.SIG_THRESHOLD:
                pred = 1
                single_p += 1
                if pred == label.detach().numpy()[0]:
                    single_tp += 1
            else:
                pred = 0
                single_n += 1
                if pred == label.detach().numpy()[0]:
                    single_tn += 1

    # confusion matrix
    tp = {'married': married_tp, 'single': single_tp}
    tn = {'married': married_tn, 'single': single_tn}
    fp = {'married': married_p - married_tp, 'single': single_p - single_tp}
    fn = {'married': married_n - married_tn, 'single': single_n - single_tn}
    Attr = ['married', 'single']
    print("tp:", tp)
    print("tn:", tn)
    print("fp:", fp)
    print("fn:", fn)

    # normal accuracy
    married_acc = (married_tp + married_tn) / (married_p + married_n)
    single_acc = (single_tp + single_tn) / (single_p + single_n)
    total_acc = (married_tp + married_tn + single_tp + single_tn) / (married_p + married_n + single_p + single_n)
    acc = {'married': married_acc, 'single': single_acc}

    # balanced accuracy: 0.5*(TP/P + TN/N)
    balanced_married_acc = 0.5 * ((married_tp / (married_n - married_tn + married_tp)) + (married_tn / (married_p - married_tp + married_tn)))
    balanced_single_acc = 0.5 * ((single_tp / (single_n - single_tn + single_tp)) + (single_tn / (single_p - single_tp + single_tn)))
    balanced_total_acc = 0.5 * (((married_tp + single_tp)/(married_n - married_tn + married_tp + single_n - single_tn + single_tp))
                                + ((married_tn + single_tn)/(married_p - married_tp + married_tn + single_p - single_tp + single_tn)))
    
    acc_b = {'married': balanced_married_acc, 'single': balanced_single_acc}

    print("married y predict accuracy/balanced married y predict accuracy:(%s/%s)" % (married_acc, balanced_married_acc))
    print("single y predict accuracy/balanced single y predict accuracy:(%s/%s)" % (single_acc, balanced_single_acc))
    print("total y predict accuracy/balanced total y predict accuracy:(%s/%s)" % (total_acc, balanced_total_acc))
    print("SP:", np.abs((married_p / (married_p + married_n)) - (single_p / (single_p + single_n))))

    # fairness calculate
    fairness_calculate(tp, tn, fn, fp, acc, acc_b, Attr)


def fairness_calculate(tp, tn, fn, fp, acc, acc_b, A):
    att_category = A
    tpr, fpr, ppv, fnr, npv, tnr = {}, {}, {}, {}, {}, {}
    # female\male fairness
    for a in att_category:
        tpr[a] = 100 * tp[a] / (tp[a] + fn[a])
        fpr[a] = 100 * fp[a] / (fp[a] + tn[a])
        ppv[a] = 100 * tp[a] / (tp[a] + fp[a])
        fnr[a] = 100 * fn[a] / (tp[a] + fn[a])
        npv[a] = 100 * tn[a] / (tn[a] + fn[a])
        tnr[a] = 100 * tn[a] / (tn[a] + fp[a])

    # predictive parity
    PP = np.abs(ppv[att_category[0]] - ppv[att_category[1]])
    # predictive equality
    PE = np.abs(fpr[att_category[0]] - fpr[att_category[1]])
    # equal opportunity
    EOp = np.abs(fnr[att_category[0]] - fnr[att_category[1]])
    # equalized odds
    EO = np.abs(fnr[att_category[0]] - fnr[att_category[1]]) + np.abs(fpr[att_category[0]] - fpr[att_category[1]])
    # conditional use accuracy equality
    CAE = np.abs(ppv[att_category[0]] - ppv[att_category[1]]) + np.abs(npv[att_category[0]] - npv[att_category[1]])
    # overall accuracy equality
    OAE = np.abs(acc[att_category[0]] - acc[att_category[1]])
    # overall accuracy equality balanced
    OAE_b = np.abs(acc_b[att_category[0]] - acc_b[att_category[1]])
    # treatment equality
    TE = np.abs(fn[att_category[0]] / fp[att_category[0]] - fn[att_category[1]] / fp[att_category[1]])

    print("PP:%s, PE:%s, EOp:%s, EO:%s, CAE:%s, TE:%s" % (PP, PE, EOp, EO, CAE, TE))
    print("OAE/balanced OAE:%s/%s" % (OAE, OAE_b))
    print("tpr:married:%s, single:%s" % (tpr[att_category[0]], tpr[att_category[1]]))
    print("tnr:married:%s, single:%s" % (tnr[att_category[0]], tnr[att_category[1]]))
    return tpr[att_category[0]]


def Equalized_odds(tp, tn, fn, fp, A):
    att_category = A
    tpr, fpr, ppv, fnr, npv, tnr = {}, {}, {}, {}, {}, {}
    # fairness items calculation
    for a in att_category:
        # tpr[a] = 100 * tp[a] / (tp[a] + fn[a])
        fpr[a] = 100 * fp[a] / (fp[a] + tn[a] + 0.00001)
        # ppv[a] = 100 * tp[a] / (tp[a] + fp[a])
        fnr[a] = 100 * fn[a] / (tp[a] + fn[a] + 0.00001)
        # npv[a] = 100 * tn[a] / (tn[a] + fn[a])
        # tnr[a] = 100 * tn[a] / (tn[a] + fp[a])

    # equalized odds
    EO = np.abs(fnr[att_category[0]] - fnr[att_category[1]]) + np.abs(fpr[att_category[0]] - fpr[att_category[1]])

    return EO


def EO_evaluation(dataset, model):
    single_tp, single_tn = 0, 0
    married_tp, married_tn = 0, 0
    single_p, single_n = 0, 0
    married_p, married_n = 0, 0
    married = torch.tensor([1, 0]).type(torch.float32)
    single = torch.tensor([0, 1]).type(torch.float32)

    for i, (data, label) in enumerate(dataset):
        data = Variable(data)
        gt = Variable(label)
        label = torch.tensor([gt[0]])
        marital = gt[1:3]
        output = model(data)
        if torch.equal(marital, married) is True:
            if output.detach().numpy()[0] >= par.SIG_THRESHOLD:
                pred = 1
                married_p += 1
                if pred == label.detach().numpy()[0]:
                    married_tp += 1
            else:
                pred = 0
                married_n += 1
                if pred == label.detach().numpy()[0]:
                    married_tn += 1
        elif torch.equal(marital, single) is True:
            if output.detach().numpy()[0] >= par.SIG_THRESHOLD:
                pred = 1
                single_p += 1
                if pred == label.detach().numpy()[0]:
                    single_tp += 1
            else:
                pred = 0
                single_n += 1
                if pred == label.detach().numpy()[0]:
                    single_tn += 1

    # confusion matrix
    tp = {'married': married_tp, 'single': single_tp}
    tn = {'married': married_tn, 'single': single_tn}
    fp = {'married': married_p - married_tp, 'single': single_p - single_tp}
    fn = {'married': married_n - married_tn, 'single': single_n - single_tn}
    Attr = ['married', 'single']
    print("tp:", tp)
    print("tn:", tn)
    print("fp:", fp)
    print("fn:", fn)

    # normal accuracy
    married_acc = (married_tp + married_tn) / (married_p + married_n)
    single_acc = (single_tp + single_tn) / (single_p + single_n)
    total_acc = (married_tp + married_tn + single_tp + single_tn) / (married_p + married_n + single_p + single_n)
    acc = {'married': married_acc, 'single': single_acc}

    # equalized odds
    EO = Equalized_odds(tp, tn, fn, fp, Attr)

    return total_acc, EO


def EO_evaluation_ae(dataset, model):
    single_tp, single_tn = 0, 0
    married_tp, married_tn = 0, 0
    single_p, single_n = 0, 0
    married_p, married_n = 0, 0
    married = torch.tensor([1, 0]).type(torch.float32)
    single = torch.tensor([0, 1]).type(torch.float32)

    for i, (data, label) in enumerate(dataset):
        data = Variable(data)
        gt = Variable(label)
        label = torch.tensor([gt[0]])
        marital = gt[1:3]
        _, _, output = model(data)
        if torch.equal(marital, married) is True:
            if output.detach().numpy()[0] >= par.SIG_THRESHOLD:
                pred = 1
                married_p += 1
                if pred == label.detach().numpy()[0]:
                    married_tp += 1
            else:
                pred = 0
                married_n += 1
                if pred == label.detach().numpy()[0]:
                    married_tn += 1
        elif torch.equal(marital, single) is True:
            if output.detach().numpy()[0] >= par.SIG_THRESHOLD:
                pred = 1
                single_p += 1
                if pred == label.detach().numpy()[0]:
                    single_tp += 1
            else:
                pred = 0
                single_n += 1
                if pred == label.detach().numpy()[0]:
                    single_tn += 1

    # confusion matrix
    tp = {'married': married_tp, 'single': single_tp}
    tn = {'married': married_tn, 'single': single_tn}
    fp = {'married': married_p - married_tp, 'single': single_p - single_tp}
    fn = {'married': married_n - married_tn, 'single': single_n - single_tn}
    Attr = ['married', 'single']
    print("tp:", tp)
    print("tn:", tn)
    print("fp:", fp)
    print("fn:", fn)

    # normal accuracy
    married_acc = (married_tp + married_tn) / (married_p + married_n)
    single_acc = (single_tp + single_tn) / (single_p + single_n)
    total_acc = (married_tp + married_tn + single_tp + single_tn) / (married_p + married_n + single_p + single_n)
    acc = {'married': married_acc, 'single': single_acc}

    # equalized odds
    EO = Equalized_odds(tp, tn, fn, fp, Attr)
    print(EO)

    return total_acc, EO


def balanced_accuracy_marital_ae(dataset, model, mode='train'):
    single_tp, single_tn = 0, 0
    married_tp, married_tn = 0, 0
    single_p, single_n = 0, 0
    married_p, married_n = 0, 0
    married = torch.tensor([1, 0]).type(torch.float32)
    single = torch.tensor([0, 1]).type(torch.float32)

    for i, (data, label) in enumerate(dataset):
        data = Variable(data)
        gt = Variable(label)
        label = torch.tensor([gt[0]])
        marital = gt[1:3]
        _, _, output = model(data)
        if torch.equal(marital, married) is True:
            if output.detach().numpy()[0] >= par.SIG_THRESHOLD:
                pred = 1
                married_p += 1
                if pred == label.detach().numpy()[0]:
                    married_tp += 1
            else:
                pred = 0
                married_n += 1
                if pred == label.detach().numpy()[0]:
                    married_tn += 1
        elif torch.equal(marital, single) is True:
            if output.detach().numpy()[0] >= par.SIG_THRESHOLD:
                pred = 1
                single_p += 1
                if pred == label.detach().numpy()[0]:
                    single_tp += 1
            else:
                pred = 0
                single_n += 1
                if pred == label.detach().numpy()[0]:
                    single_tn += 1

    # confusion matrix
    tp = {'married': married_tp, 'single': single_tp}
    tn = {'married': married_tn, 'single': single_tn}
    fp = {'married': married_p - married_tp, 'single': single_p - single_tp}
    fn = {'married': married_n - married_tn, 'single': single_n - single_tn}
    Attr = ['married', 'single']
    print("tp:", tp)
    print("tn:", tn)
    print("fp:", fp)
    print("fn:", fn)

    # normal accuracy
    married_acc = (married_tp + married_tn) / (married_p + married_n)
    single_acc = (single_tp + single_tn) / (single_p + single_n)
    total_acc = (married_tp + married_tn + single_tp + single_tn) / (married_p + married_n + single_p + single_n)
    acc = {'married': married_acc, 'single': single_acc}

    # balanced accuracy: 0.5*(TP/P + TN/N)
    balanced_married_acc = 0.5 * ((married_tp / (married_n - married_tn + married_tp)) + (
                married_tn / (married_p - married_tp + married_tn)))
    balanced_single_acc = 0.5 * (
                (single_tp / (single_n - single_tn + single_tp)) + (single_tn / (single_p - single_tp + single_tn)))
    balanced_total_acc = 0.5 * (
                ((married_tp + single_tp) / (married_n - married_tn + married_tp + single_n - single_tn + single_tp))
                + ((married_tn + single_tn) / (married_p - married_tp + married_tn + single_p - single_tp + single_tn)))

    acc_b = {'married': balanced_married_acc, 'single': balanced_single_acc}

    if mode == 'evaluate':
        print("married y predict accuracy/balanced married y predict accuracy:(%s/%s)" % (married_acc, balanced_married_acc))
        print("single y predict accuracy/balanced single y predict accuracy:(%s/%s)" % (single_acc, balanced_single_acc))
        print("total y predict accuracy/balanced total y predict accuracy:(%s/%s)" % (total_acc, balanced_total_acc))
        print("SP:", np.abs((married_p / (married_p + married_n)) - (single_p / (single_p + single_n))))

    # fairness calculate
    tpr_f = fairness_calculate(tp, tn, fn, fp, acc, acc_b, Attr)
    return tpr_f, total_acc

def evaluate(preds, label):
    preds = preds.view(preds.shape[0])
    f = (label[:, 1] == 1).sum()
    m = (label[:, 2] == 1).sum()
    p_f = ((label[:, 0] == 1) & (label[:, 1] == 1)).sum()
    p_m = ((label[:, 0] == 1) & (label[:, 2] == 1)).sum()
    n_f = f - p_f
    n_m = m - p_m
    tp_f = ((preds >= par.SIG_THRESHOLD) & (label[:, 0] == 1) & (label[:, 1] == 1)).sum()
    tp_m = ((preds >= par.SIG_THRESHOLD) & (label[:, 0] == 1) & (label[:, 2] == 1)).sum()
    fp_f = ((preds >= par.SIG_THRESHOLD) & (label[:, 0] == 0) & (label[:, 1] == 1)).sum()
    fp_m = ((preds >= par.SIG_THRESHOLD) & (label[:, 0] == 0) & (label[:, 2] == 1)).sum()
    fn_f = p_f - tp_f
    fn_m = p_m - tp_m
    tn_f = f - p_f - fp_f
    tn_m = m - p_m - fp_m

    acc = (tp_f + tp_m + tn_f + tn_m)/float(f + m)
    bacc = ((tp_f + tp_m)/float(p_f + p_m) + (tn_f + tn_m)/float(n_f + n_m))/2.0
    EO = torch.abs(fn_f/float(p_f) - fn_m/float(p_m)) + torch.abs(fp_f/float(n_f) - fp_m/float(n_m))
    tpr_f = tp_f/float(p_f)
    tpr_m = tp_m/float(p_m)
    tnr_f = tn_f/float(n_f)
    tnr_m = tn_m/float(n_m)

    return acc, bacc, EO, tpr_f, tpr_m, tnr_f, tnr_m

