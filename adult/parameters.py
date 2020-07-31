# =================== autoencoder ===================
# ratio of kld is [0,1]
RATIO_KLD = 0
RATIO_EO = 0
C_ITER = 500
NUM_WORKER = 4
# autoencoder training
AE_EPOCH = 700
AE_BATCH_SIZE = 512
SIG_THRESHOLD = 0.5
AE_LR = 0.002

FITTING_ITER = 700

PLOT_RESULT = './log/adult/model_KLD_0_%s_EO_0_%s' % (int(RATIO_KLD*10), int(RATIO_EO*10))
AE_MODEL_SAVE_PATH = './model/adult/KLD_%s_EO_%s' % (RATIO_KLD, RATIO_EO)
FITTING_MODEL = './model/adult/KLD_%s_EO_%s/KLD_AE_%s_%s.pkl' % (RATIO_KLD, RATIO_EO, FITTING_ITER, AE_EPOCH)
AE_GRAPH_SAVE_PATH = './log/graph/ae'

# encoded data
ADULT_URL = "./data/adult/adult_normalized.csv"
CLASSIFIER_TESTSET_PATH = './data/adult/encoded/adult_encoded_KLD_0_%s_EO_%s_test.csv' % (int(RATIO_KLD*10), RATIO_EO)

# bins change
KLD_BINS = 100

# visualization
VISUAL_SAVE = './log/visual/'

# testing
TESTING = False


# =================== preferential sampling ================
P_EPOCH = 50
P_LR = 0.00005
P_FITTING_ITER = 36
P_TEST_PATH = './data/adult/P/adult_preferential_KLD_0_%s_EO_0_%s_test.csv' % (int(RATIO_KLD*10), int(RATIO_EO*10))
MODEL_LOAD = './model/ae/KLD/KLD_%s_EO_%s/KLD_AE_%s_%s.pkl' % (RATIO_KLD, RATIO_EO, FITTING_ITER, AE_EPOCH)

P_PLOT = './log/P/model_KLD_0_%s_EO_0_%s' % (int(RATIO_KLD*10), int(RATIO_EO*10))
P_MODEL_SAVE = './model/P/KLD_%s_EO_%s' % (RATIO_KLD, RATIO_EO)

P_FITTING_MODEL = './model/P/KLD_%s_EO_%s/KLD_AE_%s_%s.pkl' % (RATIO_KLD, RATIO_EO, P_FITTING_ITER, P_EPOCH)



