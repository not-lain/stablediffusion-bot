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
bot = commands.Bot(command_prefix="/",intents=intents)


@bot.event
async def on_ready():
    print(f"Logged in as {bot.user} (ID: {bot.user.id})")
    synced = await bot.tree.sync()
    print(f"Synced commands: {', '.join([s.name for s in synced])}.")
    print("------")


def run_dffusion(pos_prompt: str,neg_promt: str = "", mode = "txt2img", img= None):
    """Runs the diffusion model."""
    # TODO: 
    # Add support for image prompts
    # Add support for text prompts
    print("Running diffusion")




@bot.hybrid_command(
    name="diffusion",
    description="creates an AI generated image"
)
async def diffusion(ctx, pos_prompt: str="",neg_promt: str = "", mode: str = "txt2img", img= None):
    """Creates an AI generated image based on a prompt."""
    try :
        run_dffusion(pos_prompt , neg_promt, mode, img)
        await ctx.reply("Done!")
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
event.wait()

welcome_message = """
# Welcome to the Stable Diffusion Discord Bot!

add the bot to your server by clicking [here](https://discord.com/api/oauth2/authorize?client_id=1169134134250180649&permissions=2147485696&scope=applications.commands%20bot)
"""

with gr.Blocks() as demo : 
    gr.Markdown(f"{welcome_message}")

demo.launch()

