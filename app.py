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
# from PIL import Image

dotenv.load_dotenv()
event = Event()

DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")

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


async def text2img(pos_prompt: str, neg_promt: str = ""):
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
        "stable_diffusion_checkpoint" : "3Guofeng3_v34.safetensors [50f420de]",
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
    
    
async def img2img(pos_prompt: str, neg_promt: str = "", img = None):
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
        "stable_diffusion_checkpoint" : "3Guofeng3_v34.safetensors [50f420de]",
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
    job = client.predict(*img2img_args, fn_index=1)
    
    # img = Image.open(img)
    # img.show()
    await wait(job)
    return job.result()

def run_dffusion(pos_prompt: str,neg_promt: str = "", mode = "txt2img", img= None):
    """Runs the diffusion model."""
    
    print("Running diffusion")
    
    if mode == "txt2img":
        # Support for text prompts
        generated_image = text2img(pos_prompt, neg_promt)
    elif mode == "img2img":
        # Support for image prompts
        generated_image = img2img(pos_prompt, neg_promt, img)
    else:
        return None

    return generated_image


@bot.hybrid_command(
    name="diffusion",
    description="creates an AI generated image"
)
async def diffusion(ctx, pos_prompt: str="",neg_promt: str = "", mode: str = "txt2img", img= None):
    """Creates an AI generated image based on a prompt."""
    try :
        await ctx.reply("generating ...")
        result = await run_dffusion(pos_prompt , neg_promt, mode, img)
        # wait for the result to be ready
        # print("result : ", result)
        await ctx.channel.send(file=discord.File(result))
    except Exception as e:
        print(e)
        await ctx.send("Error occured while running the model. Please try again later.")
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

add the bot to your server by clicking [here](https://discord.com/api/oauth2/authorize?client_id=1169134134250180649&permissions=2147485696&scope=applications.commands%20bot)
"""

with gr.Blocks() as demo : 
    gr.Markdown(f"{welcome_message}")

demo.queue().launch()

