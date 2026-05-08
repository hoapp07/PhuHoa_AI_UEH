
import streamlit as st
import pandas as pd
import joblib

# Load mô hình đã huấn luyện
model = joblib.load("best_model_xemay.joblib")

st.set_page_config(page_title="Dự đoán giá xe máy cũ", page_icon="🛵")
st.title("🛵 Dự đoán giá bán xe máy cũ (TP.HCM)")
st.markdown("Nhập thông tin xe để nhận giá dự đoán (triệu đồng)")

# Layout 2 cột
col1, col2 = st.columns(2)
with col1:
    hang = st.selectbox("Hãng xe", ["Honda", "Yamaha", "SYM", "Piaggio", "Suzuki", "Detech", "MOKA", "Khác"])
    dong_xe = st.text_input("Dòng xe (VD: Wave, SH, Exciter...)", "Wave")
    nam_sx = st.number_input("Năm sản xuất", min_value=1990, max_value=2026, value=2020)
    so_km = st.number_input("Số km đã chạy", min_value=0, max_value=500000, value=15000)
with col2:
    tinh_trang = st.slider("Tình trạng xe (1-10)", 1, 10, 9)
    thay_phutung = st.radio("Đã thay phụ tùng chưa?", ["No", "Yes"])
    khu_vuc = st.text_input("Khu vực bán (VD: Quận 7, Hồ Chí Minh)", "Quận 7, Hồ Chí Minh")

if st.button("🔍 Dự đoán giá"):
    # Chuẩn bị dữ liệu input
    input_data = {
        "Hãng xe": hang,
        "Dòng xe": dong_xe,
        "So_km": so_km,
        "Tinh_trang": tinh_trang * 10,   # thang 1-10 -> % (10-100)
        "Tuoi_xe": 2026 - nam_sx,
        "Khu vực bán": khu_vuc,
        "Đã thay phụ tùng chưa?": thay_phutung
    }
    input_df = pd.DataFrame([input_data])
    # Sắp xếp cột theo đúng feature_names_in_ của pipeline
    input_df = input_df[model.feature_names_in_]
    # Dự đoán
    gia = model.predict(input_df)[0]
    st.success(f"💰 Giá bán dự kiến: **{gia:,.2f} triệu đồng**")
    st.info("Lưu ý: Đây là giá tham khảo, có thể thay đổi theo tình trạng thực tế và thị trường.")