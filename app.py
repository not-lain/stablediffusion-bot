import asyncio
import os
import threading
from threading import Event
from typing import Optional
import dotenv
import discord
import gradio as gr
from discord.ext import commands
import gradio_client as grc
from gradio_client.utils import QueueError
from typing import Literal

dotenv.load_dotenv()
event = Event()

DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")

# Limited to only 25 models
models = Literal['3Guofeng3_v34.safetensors [50f420de]', 
        'absolutereality_V16.safetensors [37db0fc3]', 
        'absolutereality_v181.safetensors [3d9d4d2b]',
        'anythingV5_PrtRE.safetensors [893e49b9]', 
        'blazing_drive_v10g.safetensors [ca1c1eab]', 
        'cetusMix_Version35.safetensors [de2f2560]',
        'childrensStories_v1ToonAnime.safetensors [2ec7b88b]', 
        'Counterfeit_v30.safetensors [9e2a8f19]', 
        'cuteyukimixAdorable_midchapter3.safetensors [04bdffe6]', 
        'cyberrealistic_v33.safetensors [82b0d085]',
        'dreamlike-anime-1.0.safetensors [4520e090]', 
        'dreamlike-photoreal-2.0.safetensors [fdcf65e7]',
        'dreamshaper_8.safetensors [9d40847d]', 
        'edgeOfRealism_eorV20.safetensors [3ed5de15]', 
        'elldreths-vivid-mix.safetensors [342d9d26]', 
        'epicrealism_naturalSinRC1VAE.safetensors [90a4c676]', 
        'juggernaut_aftermath.safetensors [5e20c455]', 
        'lofi_v4.safetensors [ccc204d6]', 
        'lyriel_v16.safetensors [68fceea2]', 
        'neverendingDream_v122.safetensors [f964ceeb]', 
        'openjourney_V4.ckpt [ca2f377f]', 
        'pastelMixStylizedAnime_pruned_fp16.safetensors [793a26e8]',
        'Realistic_Vision_V5.0.safetensors [614d1063]', 
        'revAnimated_v122.safetensors [3f4fefd9]', 
        'rundiffusionFX25D_v10.safetensors [cd12b0ee]',
        ]
async def wait(job):
    while not job.done():
        await asyncio.sleep(0.2)


def get_client(session: Optional[str] = None) -> grc.Client:
    client = grc.Client("prodia/fast-stable-diffusion")
    if session:
        client.session_hash = session
    return client

intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="/",intents=intents)


@bot.event
async def on_ready():
    print(f"Logged in as {bot.user} (ID: {bot.user.id})")
    synced = await bot.tree.sync()
    print(f"Synced commands: {', '.join([s.name for s in synced])}.")
    print("------")


async def text2img(pos_prompt: str, neg_promt: str = "",model="absolutereality_v181.safetensors [3d9d4d2b]"):
    """
    Generates an image based on the given positive prompt and optional negative prompt.

    Args:
        pos_prompt (str): The positive prompt to generate the image.
        neg_prompt (str, optional): The negative prompt to generate the image. Defaults to "".

    Returns:
        The generated image.
    """
    
    txt2img_conf = {
        "parameter_11" : pos_prompt,
        "parameter_12" : neg_promt,
        "stable_diffusion_checkpoint" : model,
        "sampling_steps" : 25,
        "sampling_method" : "DPM++ 2M Karras",
        "cfg_scale" : 7,
        "width" : 512,
        "height" : 512,
        "seed" : -1
    }
    
    loop = asyncio.get_running_loop()
    client = await loop.run_in_executor(None, get_client, None)
    txt2img_args = txt2img_conf.values()
    job = client.submit(*txt2img_args, fn_index=0)
    
    # img = Image.open(img)
    # img.show()
    await wait(job)
    return job.result()
    
    
async def img2img(pos_prompt: str, neg_promt: str = "", img = None,model="absolutereality_v181.safetensors [3d9d4d2b]"):
    """
    Generates an image based on the given positive prompt, optional negative prompt and image path.

    Args:
        pos_prompt (str): The positive prompt for the image generation.
        neg_promt (str, optional): The negative prompt for the image generation. Defaults to "".
        img (filepath or URL to image): The input image for the image generation. Defaults to None.

    Returns:
        The generated image.
    """
    
    img2img_conf = {
        "parameter_52" : img,
        "denoising_strength" : 0.7,
        "parameter_44" : pos_prompt,
        "parameter_45" : neg_promt,
        "stable_diffusion_checkpoint" : model,
        "sampling_steps" : 25,
        "sampling_method" : "DPM++ 2M Karras",
        "cfg_scale" : 7,
        "width" : 512,
        "height" : 512,
        "seed" : -1
    }
    
    loop = asyncio.get_running_loop()
    client = await loop.run_in_executor(None, get_client, None)
    img2img_args = img2img_conf.values()
    job = client.submit(*img2img_args, fn_index=1)
    
    # img = Image.open(img)
    # img.show()
    await wait(job)
    return job.result()

def run_dffusion(pos_prompt: str,neg_promt: str = "",img_url= None,model="absolutereality_v181.safetensors [3d9d4d2b]"):
    """Runs the diffusion model."""

    if img_url == None:
        # Support for text prompts
        generated_image = text2img(pos_prompt, neg_promt,model)
    else:
        # Support for image prompts
        generated_image = img2img(pos_prompt, neg_promt, img_url,model)

    return generated_image


@bot.hybrid_command(
    name="diffusion",
    description="creates an AI generated image"
)
async def diffusion(ctx, pos_prompt: str="",neg_promt: str = "", img_url= None,improve : Literal["True","False"] = "True",model : models = "absolutereality_v181.safetensors [3d9d4d2b]"):
    """Creates an AI generated image based on a prompt."""
    # string injection for improved results
    _improved_pos_prompt = "ultrarealistic,8k"
    _improved_neg_promt = "3d, cartoon, (deformed eyes, nose, ears, nose), bad anatomy, ugly,blur"
    try :
        pos_prompt =pos_prompt.strip()
        neg_promt = neg_promt.strip()
        # check if the user wants to improve the results
        if improve == "True":
            if len(pos_prompt) == 0:
                pos_prompt = _improved_pos_prompt
            elif pos_prompt[-1] != ",": 
                pos_prompt += "," + _improved_pos_prompt
            else: 
                pos_prompt += _improved_pos_prompt
            if len(neg_promt) == 0:
                neg_promt = _improved_neg_promt
            elif neg_promt[-1] != ",":
                neg_promt += "," + _improved_neg_promt
            else:
                neg_promt += _improved_neg_promt
        # show the configuration
        await ctx.reply(f"""
                        generating image for { ctx.author.mention } ... \
                            ```pos_prompt: {pos_prompt} \nneg_prompt: {neg_promt}\ncontrol image_url: {img_url}\nmodel: {model}```
                            """)
        # run the API
        result = await run_dffusion(pos_prompt , neg_promt,img_url,model=model)
        await ctx.channel.send(file=discord.File(result))
    except Exception as e:
        await ctx.send(f"Error occured while running the model. Please try again later.\n{e}")
        return



def run_bot():
    try:
        bot.run(DISCORD_TOKEN)
    except Exception as e:
        print(e)
        event.set()

threading.Thread(target=run_bot).start()

welcome_message = """
# Welcome to the Stable Diffusion Discord Bot!

add the bot to your server by clicking [here](https://discord.com/api/oauth2/authorize?client_id=1169134134250180649&permissions=2147518464&scope=bot%20applications.commands)
"""

with gr.Blocks() as demo : 
    gr.Markdown(f"{welcome_message}")

demo.queue().launch()

