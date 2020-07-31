import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, balanced_accuracy_score


def calculate_fairness(id_f, id_m, Y_val, predictions):
    accuracy_f = 100 * accuracy_score(Y_val[id_f], predictions[id_f])
    accuracy_m = 100 * accuracy_score(Y_val[id_m], predictions[id_m])
    # statistical parity
    SP_f = 100 * np.where(predictions[id_f] == 1)[0].size / id_f.size
    SP_m = 100 * np.where(predictions[id_m] == 1)[0].size / id_m.size
    SP = np.abs(SP_f - SP_m)
    # tp,tn,fp,fn female
    tn_f, fp_f, fn_f, tp_f = confusion_matrix(Y_val[id_f], predictions[id_f]).ravel()
    tpr_f = 100 * tp_f / (tp_f + fn_f)
    fpr_f = 100 * fp_f / (fp_f + tn_f)
    ppv_f = 100 * tp_f / (tp_f + fp_f)
    fnr_f = 100 * fn_f / (tp_f + fn_f)
    npv_f = 100 * tn_f / (tn_f + fn_f)
    tnr_f = 100 * tn_f / (tn_f + fp_f)
    # tp,tn,fp,fn male
    tn_m, fp_m, fn_m, tp_m = confusion_matrix(Y_val[id_m], predictions[id_m]).ravel()
    tpr_m = 100 * tp_m / (tp_m + fn_m)
    fpr_m = 100 * fp_m / (fp_m + tn_m)
    ppv_m = 100 * tp_m / (tp_m + fp_m)
    fnr_m = 100 * fn_m / (tp_m + fn_m)
    npv_m = 100 * tn_m / (tn_m + fn_m)
    tnr_m = 100 * tn_m / (tn_m + fp_m)
    # predictive parity
    PP = np.abs(ppv_f - ppv_m)
    # predictive equality
    PE = np.abs(fpr_f - fpr_m)
    # equal opportunity
    EOp = np.abs(fnr_f - fnr_m)
    # equalized odds
    EO = np.abs(tpr_f - tpr_m) + np.abs(tnr_f - tnr_m)
    # conditional use accuracy equality
    CAE = np.abs(ppv_f - ppv_m) + np.abs(npv_f - npv_m)
    # overall accuracy equality
    OAE = np.abs(accuracy_f - accuracy_m)
    # Treatment equality
    TE = np.abs(fn_f / fp_f - fn_m / fp_m)

    return SP, PP, PE, EOp, EO, CAE, OAE, TE


def accuracy_fairness(models, id_f, id_m, X_train, Y_train, X_val, Y_val):
    Acc, Acc_b, SP, PP, PE, EOp, EO, CAE, OAE, TE = {},{},{},{},{},{},{},{},{},{}
    for name, model in models:
        model.fit(X_train, Y_train)
        predictions = model.predict(X_val)
        Acc[name] = 100 * accuracy_score(Y_val, predictions)
        Acc_b[name] = 100 * balanced_accuracy_score(Y_val, predictions)
        SP[name], PP[name], PE[name], EOp[name], EO[name], CAE[name], OAE[name], TE[name] = calculate_fairness(
                             id_f, id_m, Y_val, predictions)
        msg = "%s: %f, %f, %f, %f, %f, %f, %f, %f, %f, %f" % (name, Acc[name], Acc_b[name], SP[name], PP[name], PE[name],
                                            EOp[name], EO[name], CAE[name], OAE[name], TE[name])
        print(msg)
    # convert list to numpy
    Acc, Acc_b, SP, PP, EOp, EO, OAE = np.array(list(Acc.values()))[:, None], np.array(list(Acc_b.values()))[:, None], \
                                       np.array(list(SP.values()))[:, None], np.array(list(PP.values()))[:, None], \
                                       np.array(list(EOp.values()))[:, None], np.array(list(EO.values()))[:, None], \
                                       np.array(list(OAE.values()))[:, None]
    data = np.concatenate((Acc, Acc_b, SP, PP, EOp, EO, OAE), axis=1)
    print(data)
    return data, Acc, Acc_b, SP, PP, EOp, EO, OAE


# ======fairness calculation with weight======


def calculate_fairness_weight(id_f, id_m, Y_val, predictions, sample_weight):
    accuracy_f = 100 * accuracy_score(Y_val[id_f], predictions[id_f], sample_weight=sample_weight[id_f])
    accuracy_m = 100 * accuracy_score(Y_val[id_m], predictions[id_m], sample_weight=sample_weight[id_m])
    # statistical parity
    SP_f = 100 * np.where(predictions[id_f] == 1)[0].size / id_f.size
    SP_m = 100 * np.where(predictions[id_m] == 1)[0].size / id_m.size
    SP = np.abs(SP_f - SP_m)
    # tp,tn,fp,fn female
    tn_f, fp_f, fn_f, tp_f = confusion_matrix(Y_val[id_f], predictions[id_f]).ravel()
    tpr_f = 100 * tp_f / (tp_f + fn_f)
    fpr_f = 100 * fp_f / (fp_f + tn_f)
    ppv_f = 100 * tp_f / (tp_f + fp_f)
    fnr_f = 100 * fn_f / (tp_f + fn_f)
    npv_f = 100 * tn_f / (tn_f + fn_f)
    tnr_f = 100 * tn_f / (tn_f + fp_f)
    # tp,tn,fp,fn male
    tn_m, fp_m, fn_m, tp_m = confusion_matrix(Y_val[id_m], predictions[id_m]).ravel()
    tpr_m = 100 * tp_m / (tp_m + fn_m)
    fpr_m = 100 * fp_m / (fp_m + tn_m)
    ppv_m = 100 * tp_m / (tp_m + fp_m)
    fnr_m = 100 * fn_m / (tp_m + fn_m)
    npv_m = 100 * tn_m / (tn_m + fn_m)
    tnr_m = 100 * tn_m / (tn_m + fp_m)
    # predictive parity
    PP = np.abs(ppv_f - ppv_m)
    # predictive equality
    PE = np.abs(fpr_f - fpr_m)
    # equal opportunity
    EOp = np.abs(fnr_f - fnr_m)
    # equalized odds
    EO = np.abs(tpr_f - tpr_m) + np.abs(tnr_f - tnr_m)
    # conditional use accuracy equality
    CAE = np.abs(ppv_f - ppv_m) + np.abs(npv_f - npv_m)
    # overall accuracy equality
    OAE = np.abs(accuracy_f - accuracy_m)
    # Treatment equality
    TE = np.abs(fn_f / fp_f - fn_m / fp_m)

    return SP, PP, PE, EOp, EO, CAE, OAE, TE


def accuracy_fairness_reweight(models, id_f, id_m, X_train, Y_train, X_val, Y_val, X_train_au, Y_train_au, weight_train):
    Acc, SP, PP, PE, EOp, EO, CAE, OAE, TE = {},{},{},{},{},{},{},{},{}
    for name, model in models:
        if name is 'KNN' or name is 'MLP':
            model.fit(X_train_au, Y_train_au)
            predictions = model.predict(X_val)
            Acc[name]=100*accuracy_score(Y_val, predictions)
            SP[name], PP[name], PE[name], EOp[name], EO[name], CAE[name], OAE[name], TE[name] = calculate_fairness(
                                id_f, id_m, Y_val, predictions)
        else:
            model.fit(X_train, Y_train, weight_train)
            predictions = model.predict(X_val)
            Acc[name] = 100 * accuracy_score(Y_val, predictions)
            SP[name], PP[name], PE[name], EOp[name], EO[name], CAE[name], OAE[name], TE[name] = calculate_fairness(
                                id_f, id_m, Y_val, predictions)
        msg = "%s: %f, %f, %f, %f, %f, %f, %f, %f, %f" % (name, Acc[name],SP[name], PP[name], PE[name],
                                            EOp[name], EO[name], CAE[name], OAE[name], TE[name])
        print(msg)
    # convert list to numpy
    Acc, SP, PP, EOp, EO, OAE = np.array(list(Acc.values()))[:, None], np.array(list(SP.values()))[:, None],\
                                np.array(list(PP.values()))[:, None], np.array(list(EOp.values()))[:, None],\
                                np.array(list(EO.values()))[:, None], np.array(list(OAE.values()))[:, None]
    data = np.concatenate((Acc, SP, PP, EOp, EO, OAE), axis=1)
    # print(data)
    return data, Acc, SP, PP, EOp, EO, OAE

