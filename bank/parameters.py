'''
# =====================bank dataset========================
'''
# =================== autoencoder ===================
# ratio of kld is [0,1]
RATIO_KLD = 0
RATIO_EO = 0.4
C_ITER = 120
NUM_WORKER = 4
BANK_URL = './data/bank_full_normalized.csv'
########################
# autoencoder training
########################
AE_EPOCH = 700
AE_BATCH_SIZE = 512
AE_LR = 0.002
BINS = 50  # 50, 100, 200, 300, 400, 500

FITTING_ITER = 297

PLOT_RESULT = './log/bank/model_KLD_0_%s_EO_0_%s' % (int(RATIO_KLD*10), int(RATIO_EO*10))
AE_MODEL_SAVE_PATH = './model/bank/KLD_%s_EO_%s' % (RATIO_KLD, RATIO_EO)
FITTING_MODEL = './model/bank/KLD_%s_EO_%s/KLD_AE_%s_%s.pkl' % (RATIO_KLD, RATIO_EO, FITTING_ITER, AE_EPOCH)
AE_GRAPH_SAVE_PATH = './log/graph/ae'

SIG_THRESHOLD = 0.5
# visualization
VISUAL_SAVE = './log/visual/'

# testing
TESTING = False

# =================== MLP classifier ================
CLASSIFIER_TESTSET_PATH = './data/encoded/bank_encoded_KLD_0_%s_EO_%s_test.csv' % (int(RATIO_KLD*10), int(RATIO_EO*10))


# ================== preferential sampling ====================
P_EPOCH = 50
P_LR = 0.00005
P_FITTING_ITER = 20
P_TEST_PATH = './data/bank/P/adult_preferential_KLD_0_%s_EO_0_%s_test.csv' % (int(RATIO_KLD*10), int(RATIO_EO*10))
MODEL_LOAD = './model/ae/KLD/KLD_%s_EO_%s/KLD_AE_%s_%s.pkl' % (RATIO_KLD, RATIO_EO, FITTING_ITER, AE_EPOCH)

P_PLOT = './log/P/model_KLD_0_%s_EO_0_%s' % (int(RATIO_KLD*10), int(RATIO_EO*10))
P_MODEL_SAVE = './model/P/KLD_%s_EO_%s' % (RATIO_KLD, RATIO_EO)

P_FITTING_MODEL = './model/P/KLD_%s_EO_%s/KLD_AE_%s_%s.pkl' % (RATIO_KLD, RATIO_EO, P_FITTING_ITER, P_EPOCH)
