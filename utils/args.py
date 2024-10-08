import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode',               type=str,       default="")
    parser.add_argument('--sort',               type=str,       default="")
    
    parser.add_argument('--model_weights',      type=str,       default="")
    parser.add_argument('--data_path',          type=str,       default="")
    parser.add_argument('--prompt_template',    type=str,       default="")
    parser.add_argument('--log_name',           type=str,       default="")
    parser.add_argument('--epochs',             type=int,       default="")
    parser.add_argument('--learning_rate',      type=int,       default="")
    parser.add_argument('--cut_off',            type=int,       default="")    

    args = parser.parse_args()

    return args

