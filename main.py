from __future__ import print_function
import argparse
import torch
# from src.models.model import Dysarthria_model
# from src.data.make_dataset import get_train_test_val_set
# from src.train.train import train_model, test_model
from dotenv import load_dotenv
import wandb
import os
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import LoraConfig
from trl import SFTTrainer, DPOTrainer, DPOConfig

load_dotenv()

HF_token = os.getenv("HF_TOKEN")

# load_dotenv()
# WANDB_API_KEY = os.getenv("WANDB_API_KEY")



def main():
    # # Training settings
    # parser = argparse.ArgumentParser(description='Dysarthria classification training')

    # parser.add_argument('--wandb_token', type=str ,help='Weights and Biases Token')

    # parser.add_argument('--batch_size', type=int, default=8, metavar='N',
    #                     help='input batch size for training (default: 64)')
    
    # parser.add_argument('--epochs', type=int, default=80, metavar='N',
    #                     help='number of epochs to train (default: 14)')
    
    # parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
    #                     help='learning rate (default: 1e-4)')

    # parser.add_argument('--SSL_model', type=str, default="facebook/hubert-base-ls960", 
    #                     help='Huggingface model path')

    # parser.add_argument('--SSL_model_name', default='Hubert', const='Hubert', nargs='?',  choices=["Hubert", "Wav2Vec2"],
    #                     help='Model name')
    
    # parser.add_argument('--data_path', type=str, default="data/raw" ,help='path to the audio files')

    # parser.add_argument('--data_splits_path', type=str, default="./data/data_splits" ,help='path to the data CSV files containing data splits')
    
    # parser.add_argument('--output_path', type=str, default="./models" ,help='path to the output dir')

    # parser.add_argument('--wandb_project_name', type=str, default="Dysarthria Regression", help='wandb project name')

    # parser.add_argument('--wandb_run_name', type=str, default="Test Run",  help='current wandb run name')

    # parser.add_argument('--EXP_name', type=str , default="Reg Test Run", help='the name of the experiment examples: EXP1, EXP2, ...')


    # parser.add_argument('--no_cuda', action='store_true', default=False,
    #                     help='disables CUDA training')
    
    # parser.add_argument('--no_mps', action='store_true', default=False,
    #                     help='disables macOS GPU training')
    
    # parser.add_argument('--dry_run', action='store_true', default=False,
    #                     help='quickly check a single pass')
    
    # parser.add_argument('--seed', type=int, default=101, metavar='S',
    #                     help='random seed (default: 101)')
    

    # parser.add_argument('--training_scheme', default='pt', const='pt', nargs='?', choices=['frozen', 'pt', 'ft'],help='freeze the SSL model, partial trainging (pt) by freezing the CNN layers(feature extractor), full training (ft) by training the whole model')


    # parser.add_argument('--continue_training', action='store_true', default=False,
    #                     help='continue training from a checkpoint')
    
    # parser.add_argument('--cross_validation', action='store_true', default=False,
    #                     help='use 5-folds cross validation')

    # parser.add_argument('--checkpoint_path', type=str, default="./models" ,help='path to the checkpoint')

    


    
    
    


    
    # args = parser.parse_args()
    # use_cuda = not args.no_cuda and torch.cuda.is_available()
    # use_mps = not args.no_mps and torch.backends.mps.is_available()


    torch.manual_seed(101)

    print("checking available device...")
    print(f" CUDA: {torch.cuda.is_available()} \n Number Of Devices: {torch.cuda.device_count()} \n Curr device id: {torch.cuda.current_device()} \n Device name: {torch.cuda.get_device_name(0)}")

    # if use_cuda:
    #     device = torch.device("cuda")
    # elif use_mps:
    #     device = torch.device("mps")
    # else:
    #     device = torch.device("cpu")

    # print("device: ", device)

    print("define the model...")
    compute_dtype = getattr(torch, "float16")

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=False,
    )
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B", token= HF_token)
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B", device_map={"": 0}, quantization_config=quant_config, token=HF_token)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # model = Dysarthria_model(args.SSL_model).to(device)

    # if args.training_scheme == "frozen":
    #     model.freeze_whole_SSL_model()

    # elif args.training_scheme == "pt":
    #     model.freeze_feature_extractor()


    # if args.continue_training:
    #     print("loading checkpoint...")
    #     checkpoint = model.load_state_dict(torch.load(args.checkpoint_path))
    #     checkpoint = torch.load(torch.load(args.checkpoint_path))
    #     model.load_state_dict(checkpoint['model_state_dict'])
    #     args.wandb_run_id = checkpoint['run Id']
    #     args.wandb_run_name = checkpoint['run name']

    print("download and prepare the dataset...")
    anthropicHH = "trl-internal-testing/hh-rlhf-trl-style"
    anthropicHH_dataset_train = load_dataset(anthropicHH, split="train")
    anthropicHH_dataset_test = load_dataset(anthropicHH, split="test")
    def process(row):
        row["chosen"] = tokenizer.apply_chat_template(row["chosen"], tokenize=False)
        row["rejected"] = tokenizer.apply_chat_template(row["rejected"], tokenize=False)
        return row

    anthropicHH_dataset_train = anthropicHH_dataset_train.map(
        process,
        # num_proc=multiprocessing.cpu_count(),
        load_from_cache_file=False,
    )
    anthropicHH_dataset_test = anthropicHH_dataset_test.map(
        process,
        # num_proc=multiprocessing.cpu_count(),
        load_from_cache_file=False,
    )

    print("defining DPO trainer...")
    peft_params = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=64,
        bias="none",
        task_type="CAUSAL_LM",
    )

    training_args = DPOConfig(
        beta=0.1,
        output_dir="./output",
    )

    trainer = DPOTrainer(
        model,
        ref_model=None,
        args=training_args,
        train_dataset=anthropicHH_dataset_train,
        eval_dataset=anthropicHH_dataset_test,
        tokenizer=tokenizer,
        # peft_config=get_peft_config(model_config),
        peft_config=peft_params,
        # callbacks=[RichProgressCallback] if TRL_USE_RICH else None,
    )


#     print("defining loss and optimizer..")
#     loss_fn = torch.nn.MSELoss()

#     optimizer = torch.optim.SGD(model.parameters(), lr= args.lr)

#     # if args.cross_validation:
#     #     for i in range(5):
#     #         print(f"start training fold leaving fold {i}...")
#     #         init_wandb_run(args, fold = i)
#     #         train_set, test_set = get_train_test_set(args, args.folds, args.SSL_model, i)
#     #         train_model(args, model, train_set, test_set, loss_fn, optimizer, args.epochs, args.batch_size, device, args.checkpoint_path, args.continue_training, wandb)
#     #         test_model(args, model, test_set, loss_fn, device, wandb)
    
#     # else:
#         # init_wandb_run(args)
#         # train_set, test_set = get_train_test_set(args, 4)
#         # output_path = os.path.join(args.output_path, f"{args.EXP_name}_{args.SSL_model_name}.pt")
#         # train_model(model, train_set, test_set, loss_fn, optimizer, args.epochs, args.batch_size, device, output_path, args.checkpoint_path, args.continue_training, wandb)
#         # test_model(model, test_set, loss_fn, device, wandb)
    
#     init_wandb_run(args)
#     train_dataloader, test_dataloader, val_dataloader = get_train_test_val_set(args)
#     output_path = os.path.join(args.output_path, f"{args.EXP_name}_{args.SSL_model_name}.pt")
#     train_model(args, model, train_dataloader, val_dataloader, optimizer, loss_fn, device, output_path = output_path, wandb=wandb)
#     test_model(model, test_dataloader, loss_fn, device, wandb=wandb)

# wandb.finish()
    


# def init_wandb_run(args, fold = None):

#     print("init wandb run...")
#     # wandb.login(key = WANDB_API_KEY)
#     wandb.login(key = args.wandb_token)

#     if fold:
#         wandb_run_name = f"{args.wandb_run_name}_fold{fold}"
#     else:
#         wandb_run_name = args.wandb_run_name

#     if args.continue_training:
#         print(f"continue {wandb_run_name} wandb run...")
#         wandb.init(
#             # set the wandb project where this run will be logged
#             project= args.wandb_project_name,
#             id = args.wandb_run_id,
#             resume="must",
#             # name = args.wandb_run_name,

#             # track hyperparameters and run metadata
#             config={
#             "learning_rate": args.lr,
#             "architecture": args.SSL_model_name,
#             "dataset": "UASpeech",
#             "epochs": args.epochs,
#         })
#     else:
#         print(f"start {wandb_run_name} wandb run...")
#         wandb.init(
#             # set the wandb project where this run will be logged
#             project= args.wandb_project_name,
#             # id="pwim79rh",
#             # resume="must",
#             name = args.wandb_run_name,

#             # track hyperparameters and run metadata
#             config={
#             "learning_rate": args.lr,
#             "architecture": args.SSL_model_name,
#             "dataset": "UASpeech",
#             "epochs": args.epochs,
#         })

    



    # if args.save_model:
    #     torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == '__main__':
    main()