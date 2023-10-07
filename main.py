import gradio as gr
import os
import json
import os

def main():
    HF_DIR = '/.huggingface'
    MODEL_NAME = "stabilityai/stable-diffusion-2-1"
    OUTPUT_DIR = "/output"
    CONCEPTS_DIR = "content/data"
    if not os.path.exists(HF_DIR):
        os.mkdir(HF_DIR) 
    if not os.path.exists(OUTPUT_DIR):
        os.mkdir(OUTPUT_DIR) 
    if not os.path.exists(CONCEPTS_DIR):
        os.mkdir(OUTPUT_DIR + "/instance")
        os.mkdir(OUTPUT_DIR + "/class") 

    HUGGINGFACE_TOKEN = "" #@param {type:"string"}
    print(f"{HUGGINGFACE_TOKEN} > ~/.huggingface/token")


    with gr.Blocks() as demo:
        gr.Markdown("Start typing below and then click **Run** to see the output.")
        with gr.Row():
            train_instance = gr.Textbox(label="Instance")
            train_class = gr.Textbox(label="Class")
        with gr.Row():
            gr.Image(label="Train Image(s)" )
        btn = gr.Button("Run")
        btn.click(fn=run, inputs=[train_instance,train_class])
            
    demo.launch()

def run(train_instance,train_class):
    concepts_list = [
        {
            "instance_prompt":     train_instance,
            "class_prompt":        train_class,
            "instance_data_dir":    "content/data/instance",
            "class_data_dir":       "content/data/class"
        }
    ]

    for c in concepts_list:
        os.makedirs(c["instance_data_dir"], exist_ok=True)

    with open("concepts_list.json", "w") as f:
        json.dump(concepts_list, f, indent=4)

    os.system('train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --pretrained_vae_name_or_path="stabilityai/sd-vae-ft-mse" \
  --output_dir=$OUTPUT_DIR \
  --with_prior_preservation --prior_loss_weight=1.0 \
  --seed=1337 \
  --resolution=512 \
  --train_batch_size=1 \
  --train_text_encoder \
  --mixed_precision="fp16" \
  --use_8bit_adam \
  --gradient_accumulation_steps=1 \
  --learning_rate=2e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --num_class_images=50 \
  --max_train_steps=600 \
  --save_interval=10000 \
  --concepts_list="concepts_list.json"') 
    
if __name__ == '__main__':
    main()
