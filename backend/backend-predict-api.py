import os
import torch
from PIL import Image
from torch.autograd import Variable
from torchvision import datasets, transforms
import json
import redis
from model.RANet_basic import Net
from cam_predict import cam_api

from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.docs import get_swagger_ui_html, get_redoc_html
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse


app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000",
    "http://localhost:3001",
    "http://1.15.55.32:3001",
    "http://192.168.0.109:3001"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 构造模型名称和对应路径的字典
param_dict = {
    'resnet18': 'your model pth path',
    'vgg16': 'your model pth path',
    'vgg19': 'your model pth path',
    'san10': 'your model pth path',
    'san15': 'your model pth path',
    'vgg16_se': 'your model pth path',
    'vgg16_cbam': 'your model pth path',
    'RANet': 'your model pth path',
    'LA-RANet': 'your model pth path'
}

class_dict = {
        '[0]': "平滑肌瘤",
        '[1]': "神经内分泌肿瘤",
        '[2]': "胃肠道间质瘤",
        '[3]': "异位胰腺",
        '[4]': "脂肪瘤"
        }


# 定义模型名称和对应网络结构的映射
model_dict = {
    # 'RANet': lambda: Net(5),
    'LA-RANet': lambda: Net(5)
}

device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")


def model_choose(name, device):
    # 根据名称获取对应的网络结构
    net = model_dict[name]()
    params = torch.load(param_dict[name])
    model = load_model(net, params, device)
    return model


def load_model(model, state_dict, device):
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def inference(img, img_path, model, device, transform):
    img = transform(img)
    img = torch.unsqueeze(img, 0)

    model.eval()
    img = Variable(img.to(device))
    outputs = model(img)
    probs = torch.nn.functional.softmax(outputs, dim=1)
    pred_probs, pred_class = torch.max(probs, dim=1)
    pred_probs = pred_probs.detach().cpu().numpy()
    pred_class = pred_class.detach().cpu().numpy()

    print("预测类别：", pred_class,"预测置信度：", pred_probs[0])

    # create Grad-CAM
    save_path = "./upload"
    os.makedirs(save_path, exist_ok=True)
    save_name = os.path.basename(img_path)
    cam_api(img_path, model, save_name)

    return pred_class, pred_probs[0], save_name


@app.post("/predict/")
async def predict(img_path: str = File(...)):
    print("接受的图像路径：", img_path)
    print(os.getcwd())
    name = 'vgg16_cbam'
    gray_model = ['resnet18', 'vgg19', 'san10', 'san15', 'apcnn']
    model = model_choose(name, device)

    # 定义Transforms
    if name in gray_model:
        img = Image.open(img_path)
        print('图像尺寸: ', img.size)
        mean, std = [0.5], [0.5]
        transform = transforms.Compose([transforms.Resize(256),
                                              transforms.CenterCrop(224),
                                              transforms.Grayscale(num_output_channels=1),
                                              transforms.ToTensor(),
                                              transforms.Normalize(mean, std)])
    
    else:
        img = Image.open(img_path).convert('RGB')
        print('图像尺寸: ', img.size)
        mean, std = [0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]
        transform = transforms.Compose([transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean, std)])

    predicted_cls, predicted_probs, save_name = inference(img, img_path, model, device, transform)

    json_content = {"cls": class_dict[str(predicted_cls)], "probs": str(predicted_probs),"url": f"http://your_ip_address:8002/upload/cam_{save_name}"}
    return JSONResponse(content=json_content)
    

# Add Swagger UI and ReDoc routes
@app.get("/docs", include_in_schema=False)
async def docs():
    return get_swagger_ui_html(openapi_url="/openapi.json", title="API Docs")

@app.get("/redoc", include_in_schema=False)
async def redoc():
    return get_redoc_html(openapi_url="/openapi.json", title="API Docs")

# Serve openapi.json file
@app.get("/openapi.json", include_in_schema=False)
async def get_open_api_endpoint():
    return app.openapi()


# 创建 Redis 客户端连接
redis_client = redis.Redis(host='127.0.0.1', port=6379, db=0)


# 上传图像
@app.post("/upload/")
async def create_upload_file(image: UploadFile = File(...)):
    filename = image.filename
    with open(f"./upload/{filename}", "wb") as f:
        f.write(await image.read())

    filepath = os.path.abspath(f"./upload/{filename}")
    url = f"http://1.15.55.32:8002/upload/{filename}"
    
    # 构建图像的 URL 地址，并缓存到 Redis 中
    redis_client.lpush('image_urls', url)

    return {"filename": filename, "url": url, "filepath": filepath}


@app.post("/report")
async def upload(request: Request):
    form = await request.form()
    img_name = form['img_name']
    with open(f"reports/{img_name}.json", "w") as f:
        json.dump(dict(form), f)
    return {"msg": "data saved successfully"}



# 附加指定路径下的静态文件
app.mount("/upload", StaticFiles(directory="upload"), name="upload")


# redis
# 定义获取图像 URL 列表的 API 接口
@app.get("/get_image_urls")
async def get_image_urls():
    # 创建 Redis 客户端连接
    
    # 从 Redis 中获取图像 URL 列表，并返回给前端
    image_urls = redis_client.lrange('image_urls', 0, -1)
    return {"urls": [url.decode('utf-8') for url in image_urls]}


