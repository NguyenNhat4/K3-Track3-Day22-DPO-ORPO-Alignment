# Preference Alignment Lab Guide

Chào mừng bạn đến với bài lab về **DPO (Direct Preference Optimization)** và **ORPO (Odds Ratio Preference Optimization)** Alignment.

> 📖 **Xem Tài Liệu Kỹ Thuật Chi Tiết**: Để xem hướng dẫn từng bước từ A-Z, giải thích toán học chi tiết, minh họa code và mẹo debugging cho người mới bắt đầu, hãy mở file [TECHNICAL_DOCS.md](file:///home/nhatnm/code/vin-project/lab/phrase2/K3-Track3-Day22-DPO-ORPO-Alignment/docs/TECHNICAL_DOCS.md).

---

## Các Nhiệm Vụ Cần Hoàn Thành (Milestones)

### 📌 Task 1: Data Loader & Schema Validation (30 min)
- Sửa hàm validation trong `src/preference_lab/schemas.py`.
- Bổ sung đếm dòng và thông báo lỗi chi tiết khi đọc file JSONL trong `src/preference_lab/data.py`.
- Thực hiện chia tập Train/Val theo Prompt (`split_by_prompt`) tránh Data Leakage.
- Kiểm tra bằng lệnh: `pytest tests/test_data.py`

### 📌 Task 1.5: (Tùy chọn) Synthetic Data Generation (20 min)
- Mở rộng tập dữ liệu bằng OpenAI API qua script `scripts/generate_data.py`:
  ```bash
  export OPENAI_API_KEY=your_key
  python scripts/generate_data.py --count 5 --domain "python coding"
  ```

### 📌 Task 2: Cài Đặt DPO hoặc ORPO Loss (30 min)
- Lựa chọn DPO hoặc ORPO và hoàn thành các `TODO(student)` trong `src/preference_lab/losses.py`.
- Đảm bảo tính ổn định số học (sử dụng Log-Sigmoid và `np.log1p`).
- Kiểm tra bằng lệnh: `pytest tests/test_losses.py`

### 📌 Task 3: Evaluation & Pairwise Accuracy (15 min)
- Cài đặt hàm `pairwise_accuracy` trong `src/preference_lab/evaluate.py`.
- Chạy lệnh CLI để xuất metrics:
  ```bash
  pref-lab evaluate --config configs/local.yaml
  ```
- Kết quả lưu tại: `outputs/metrics.json`

### 📌 Task 4: Safety Regression & Report (15 min)
- Kiểm thử độ an toàn trên các prompt trong `docs/regression_prompts.md`.
- Điền đầy đủ thông tin báo cáo thu hoạch vào `docs/REPORT_TEMPLATE.md`.

