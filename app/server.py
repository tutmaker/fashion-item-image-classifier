import aiohttp
import asyncio
import uvicorn
from fastai import *
from fastai.vision import *
from io import BytesIO
from starlette.applications import Starlette
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import HTMLResponse, JSONResponse
from starlette.staticfiles import StaticFiles

export_file_url = 'https://drive.google.com/uc?export=download&id=1XzEApsi4D9Zvgx6XhQ_xaqjfiCbHWscA'
export_file_name = 'fashion-classifier.pkl'

classes = classes = [['Accessory Gift Set','Baby Dolls','Backpacks','Bangle','Basketballs','Bath Robe','Belts','Booties','Boxers','Bra','Bracelet','Briefs','Camisoles','Capris','Caps','Casual Shoes','Churidar','Clutches','Compact','Concealer','Cufflinks','Deodorant','Dresses','Duffel Bag','Dupatta','Earrings','Eyeshadow','Face Moisturisers','Face Wash and Cleanser','Flats','Flip Flops','Formal Shoes','Foundation and Primer','Fragrance Gift Set','Free Gifts','Gloves','Hair Colour','Handbags','Heels','Highlighter and Blush','Innerwear Vests','Jackets','Jeans','Jeggings','Jewellery Set','Jumpsuit','Kajal and Eyeliner','Kurta Sets','Kurtas','Kurtis','Laptop Bag','Leggings','Lip Care','Lip Gloss','Lip Liner','Lipstick','Lounge Pants','Lounge Shorts','Mascara','Mask and Peel','Messenger Bag','Mobile Pouch','Mufflers','Nail Polish','Necklace and Chains','Night suits','Nightdress','Patiala','Pendant','Perfume and Body Mist','Rain Jacket','Ring','Rompers','Rucksacks','Salwar','Sandals','Sarees','Scarves','Shirts','Shoe Accessories','Shorts','Skirts','Socks','Sports Sandals','Sports Shoes','Stockings','Stoles','Sunglasses','Sunscreen','Suspenders','Sweaters','Sweatshirts','Swimwear','Ties','Tops','Track Pants','Tracksuits','Travel Accessory','Trousers','Trunk','Tshirts','Tunics','Waist Pouch','Waistcoat','Wallets','Watches','Water Bottle']]

path = Path(__file__).parent

app = Starlette()
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_headers=['X-Requested-With', 'Content-Type'])
app.mount('/static', StaticFiles(directory='app/static'))


async def download_file(url, dest):
    if dest.exists(): return
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            data = await response.read()
            with open(dest, 'wb') as f:
                f.write(data)


async def setup_learner():
    await download_file(export_file_url, path / export_file_name)
    try:
        learn = load_learner(path, export_file_name)
        return learn
    except RuntimeError as e:
        if len(e.args) > 0 and 'CPU-only machine' in e.args[0]:
            print(e)
            message = "\n\nThis model was trained with an old version of fastai and will not work in a CPU environment.\n\nPlease update the fastai library in your training environment and export your model again.\n\nSee instructions for 'Returning to work' at https://course.fast.ai."
            raise RuntimeError(message)
        else:
            raise


loop = asyncio.get_event_loop()
tasks = [asyncio.ensure_future(setup_learner())]
learn = loop.run_until_complete(asyncio.gather(*tasks))[0]
loop.close()


@app.route('/')
async def homepage(request):
    html_file = path / 'view' / 'index.html'
    return HTMLResponse(html_file.open().read())


@app.route('/analyze', methods=['POST'])
async def analyze(request):
    img_data = await request.form()
    img_bytes = await (img_data['file'].read())
    img = open_image(BytesIO(img_bytes))
    prediction = learn.predict(img)[0]
    return JSONResponse({'result': str(prediction)})


if __name__ == '__main__':
    if 'serve' in sys.argv:
        uvicorn.run(app=app, host='0.0.0.0', port=5000, log_level="info")
