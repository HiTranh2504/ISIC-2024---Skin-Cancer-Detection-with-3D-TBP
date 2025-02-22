import streamlit as st 
import pandas as pd
import numpy as np
import torch
from torchvision import transforms, models
from PIL import Image

st.set_page_config(
    page_title="ISIC-2024", 
    page_icon="🧑‍💻"
)

df = pd.read_csv('train-metadata.csv')
st.session_state["df"] = df

# st.write(df[df['isic_id'] == "ISIC_0015670"]['target'][0])

# streamlit run app.py --server.enableXsrfProtection false

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

def load_model(model_path):
    model = models.efficientnet_b0(weights=None)  # Sử dụng EfficientNet-B0
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 2)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model, device


def transform_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    return transform(image).unsqueeze(0)

# Hàm để dự đoán lớp
def predict(model, device, image):
    with torch.no_grad():
        image = image.to(device)
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
    return predicted.item(), probabilities

# Tải mô hình
model, device = load_model('efficientnet_b0_skin_cancer_model.pth')

# Streamlit
st.title('MÔN HỌC: TRÍ TUỆ NHÂN TẠO')
st.title('Project: ISIC 2024 - Skin Cancer Detection with 3D-TBP')
st.write('Hey Bro, Bro có sợ bị ung thư da không, chắc Bro đang hoảng sợ và nhớ lại xem trên người mình có vết thương hay khối hắc tố da nào không chứ gì. Nhưng hãy yên tâm đi, chung tôi ở đây để giúp Bro đó.')
st.write('Nào hãy chụp một tấm ảnh trên da của Bro và đưa vào mô hình của chúng tôi để kiểm tra xem Bro có bị UNG THƯ DA không nhé.')
st.write('Lưu ý: Kết quả này không hoàn toàn chính xác 100% nhưng nếu kết quả là ÁC TÍNH có xác suất cao thì Bro nên tìm gặp bác sĩ ngay đi nhé (bác sĩ da liễu not BÁC SĨ HẢI)')

uploaded_file = st.file_uploader("tải lên đây 1 tấm ảnh đi nào: ", type=["jpg", "jpeg", "png"])
st.session_state["file"] = uploaded_file

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    st.image(image, caption='Uploaded Image', use_column_width=True, width=200)
    
    image_tensor = transform_image(image)
    
    label, probabilities = predict(model, device, image_tensor)
    
    labels = ['Non-Melanoma', 'Melanoma']
    st.write(f'Prediction: {labels[label]}')
    st.write(f'probability: {probabilities[0][label].item() * 100:.2f}%')
    
    if labels[label] == 'Melanoma':
        st.warning('Kết quả: Ác tính, Chúc Bro có một chuyến đi đến bệnh viên vui vẻ nhaaaa')
    else:
        st.warning('Kết quả: Không ác tính, tadaaa chúc mừng Bro đã không sao, nhóm em mong được thầy chấm điểm cao nhất có thể ạ !!!')

        # st.warning('Kết quả: Không ác tính, tadaaa chúc mừng Bro đã không sao, hãy vote cho đề tài tụi tui ở vị trí cao nhất nhé')
    
    _id = str(st.session_state["file"].name).split('.')[0]
    df = st.session_state['df']
    st.write('Đây là kết quả thực tế từ file CSV:')
    st.write(df[df['isic_id'] == _id][['isic_id', 'target']])