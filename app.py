import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model và bộ mã hóa
model = joblib.load('best_model.pkl')
encoder = joblib.load('encoder.pkl')
scaler = joblib.load('scaler.pkl')

st.set_page_config(page_title="Dự đoán giá xe máy cũ", page_icon="🏍️")
st.title("🏍️ Dự đoán giá bán xe máy cũ")
st.markdown("Nhập thông tin xe để nhận giá dự đoán.")

with st.form("input_form"):
    col1, col2 = st.columns(2)
    with col1:
        hang_xe = st.selectbox("Hãng xe", ['Honda', 'Yamaha', 'SYM', 'Suzuki', 'Piaggio', 'Detech', 'Moka', 'Peugeot'])
        dong_xe = st.text_input("Dòng xe (VD: Vision, Exciter...)", "Vision")
        nam_sx = st.number_input("Năm sản xuất", 1990, 2026, 2020)
    with col2:
        odo = st.number_input("Số km đã chạy (ODO)", 0, 500000, 15000)
        tinh_trang = st.slider("Tình trạng xe (%)", 0, 100, 90)
        khu_vuc = st.selectbox("Khu vực bán", [
            'Quận 1', 'Quận 2', 'Quận 3', 'Quận 4', 'Quận 5', 'Quận 6', 'Quận 7', 'Quận 8', 
            'Quận 9', 'Quận 10', 'Quận 11', 'Quận 12', 'Quận Bình Tân', 'Quận Bình Thạnh',
            'Quận Gò Vấp', 'Quận Phú Nhuận', 'Quận Tân Bình', 'Quận Tân Phú',
            'Huyện Bình Chánh', 'Huyện Củ Chi', 'Huyện Hóc Môn', 'Huyện Nhà Bè', 'TP Thủ Đức'
        ])
        thay_phu_tung = st.radio("Đã thay phụ tùng?", ['Chưa thay', 'Đã thay'])
    
    submit = st.form_submit_button("🔮 Dự đoán giá")

if submit:
    # Tạo DataFrame từ input
    input_df = pd.DataFrame({
        'Hang_xe': [hang_xe],
        'Dong_xe': [dong_xe],
        'ODO': [odo],
        'Khu_vuc': [khu_vuc],
        'Tinh_trang': [tinh_trang],
        'Thay_phu_tung': [thay_phu_tung],
        'Tuoi_xe': [2026 - nam_sx]
    })

    # Mã hóa
    cat_cols = ['Hang_xe', 'Dong_xe', 'Khu_vuc', 'Thay_phu_tung']
    input_df[cat_cols] = encoder.transform(input_df[cat_cols])

    # Scale
    num_cols = ['ODO', 'Tinh_trang', 'Tuoi_xe']
    input_df[num_cols] = scaler.transform(input_df[num_cols])

    # Dự đoán
    gia_du_doan = model.predict(input_df)[0]
    st.success(f"💰 Giá bán dự đoán: **{gia_du_doan:,.0f} VND**")
    st.caption("Dự đoán dựa trên dữ liệu thị trường, chỉ mang tính tham khảo.")