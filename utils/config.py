ffpp_original_path = "../faceforensics++/original_sequences/youtube"
ffpp_fake_path = "../faceforensics++/manipulated_sequences"
celebdf_path = "../datasets/celedf/"
dfdc_path = "../datasets/deepfake/train_faces"

DETERMINSTIC = True
RNG_SEED = 20
batch_size = 128
frame_nums = 10

compress = 'c23'    #choose c23 or c40

input_size = 224

detect_name = "mtcnn"
model_name = 'efficientnet-b4'
type_list = ['Deepfakes','Face2Face','FaceSwap','NeuralTextures','all']
type_list_short = ['df','f2f','fs','nt','all']
real_weight = 1


model_path = None
fnet_path = None

beta1 = 0.9
beta2 = 0.999
learning_rate = 0.001
metalr = 0.001
plr = 0.001
epochs = 200
step_size = 5
gamma = 0.1
weight_bce = 0.05
weight_ct = 1
# Misc
print_interval = 100
save_interval = 5
test_interval = 1
finetune_interval = 2
parallel = False
lamda = 0.01
alpha = 1
des = "train"
save_dir = f"result/output_{compress}_GCD_ablation/{compress}_{model_name}_{input_size}_{lamda}_{alpha}_{des}"