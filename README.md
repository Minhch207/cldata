
# Clata

Cldata là một cli app đơn giản vibecode bằng python với mục đích là clean data một cách nhanh chóng.



## 💾 Installation:

Để sử dụng ta git clone ứng dụng về máy

```bash
  git clone https://github.com/Minhch207/cldata.git
  cd cldata
```
Sau đó, ta có thể chạy bằng python 
```bash
   python cldata.py "heart.csv" 
```
## Requirements
- Python 3.12+
- Pandas 3.0.3+
- scikit-learn 1.8.0+

```bash
pip install pandas scikit-learn
```
## Usage/Examples
### SYNOPSIS
```bash
python cldata.py [options] [train% test% vali%] "path_to_file"
```
### [options]
- -sc  chuẩn hoá dữ liệu bằng z-score, mặc định không chuẩn hóa.
- -0en không encoding data, mặc định có encoding. 
### [train% test% vali%]
- Tỷ lệ từng phần dữ liệu một, khi dùng không cần viết %, chỉ cần viết ba số lần lượt cho train, test, validation. 
- Phần validation có thể bỏ qua và viết bằng lần lượt hai số đại diện cho test và train.
- Nếu không cho, chương trình sẽ dùng khung 80% train 20% test.
### Examples
```bash
python cldata.py -sc 70 15 15 "heart.csv" 
#chia data thành ba phần 70% train 15% test 15% validation với chuẩn hóa Z-score và có encoding.

python cldata.py -0en 80 20 "heart.csv"
#chia data thành ba phần 80% train 20% test với không encoding và không chuẩn hóa Z-score.

python cldata.py "heart.csv"
#chia data thành ba phần 80% train 20% test với có encoding và không chuẩn hóa Z-score.
```