import argparse
import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

def validate_ratios(x, y, z):
    """Kiểm tra tổng tỉ lệ có bằng 100 hay không."""
    if x + y + z != 100:
        print(f"Lỗi: Tổng tỉ lệ train ({x}) + test ({y}) + validation ({z}) = {x+y+z}. Yêu cầu tổng phải bằng 100!")
        sys.exit(1)

def clean_and_split_data(args):
    # 1. Kiểm tra định dạng file
    if not args.path_to_csv_file.lower().endswith('.csv'):
        print("Lỗi: Ứng dụng chỉ chấp nhận file định dạng .csv!")
        sys.exit(1)
        
    if not os.path.exists(args.path_to_csv_file):
        print(f"Lỗi: Không tìm thấy file tại đường dẫn '{args.path_to_csv_file}'")
        sys.exit(1)

    # Đọc dữ liệu
    print(f"--> Đang đọc file: {args.path_to_csv_file}")
    df = pd.read_csv(args.path_to_csv_file)

    # 2. Xử lý Encoding (Mặc định là CÓ encoding, trừ khi có flag --no-en)
    if args.encoding:
        print("--> Đang thực hiện Encoding các cột dữ liệu không phải số...")
        # Lấy các cột có kiểu dữ liệu là object hoặc category (chuỗi/chữ)
        categorical_cols = df.select_dtypes(include=['object', 'category', 'str']).columns
        
        for col in categorical_cols:
            le = LabelEncoder()
            # Xử lý các giá trị NaN/Null trước khi encode (nếu có) thành chuỗi để tránh lỗi
            df[col] = df[col].astype(str)
            df[col] = le.fit_transform(df[col])
    else:
        print("--> Bỏ qua bước Encoding theo yêu cầu (-0en).")

    # 3. Xử lý Scale dữ liệu bằng Z-score (Mặc định là KHÔNG, chỉ làm nếu có flag -sc)
    if args.scale:
        print("--> Đang chuẩn hóa dữ liệu bằng Z-score...")
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        if len(numeric_cols) > 0:
            scaler = StandardScaler()
            df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
        else:
            print("Cảnh báo: Không có cột số nào để chuẩn hóa Z-score.")
    else:
        print("--> Bỏ qua bước chuẩn hóa Z-score.")

    # 4. Chia dữ liệu ngẫu nhiên (Train, Test, Validation)
    x, y, z = args.ratios[0], args.ratios[1], args.ratios[2]
    validate_ratios(x, y, z)
    
    print(f"--> Đang chia dữ liệu theo tỉ lệ: Train ({x}%), Test ({y}%), Validation ({z}%)")
    
    # Đổi tỉ lệ từ phần trăm sang số thập phân [0, 1]
    test_size = y / 100.0
    val_size = z / 100.0
    
    # Khởi tạo các dataframe trống
    train_df, test_df, val_df = None, None, None
    
    # Xáo trộn và chia dữ liệu ngẫu nhiên (sử dụng random_state để đảm bảo tính nhất quán nếu cần, bỏ ra nếu muốn ngẫu nhiên hoàn toàn)
    if z > 0:
        # Nếu có validation, chia lần 1 tách lấy Train, phần còn lại là (Test + Val)
        remaining_size = test_size + val_size
        train_df, remaining_df = train_test_split(df, test_size=remaining_size, random_state=42, shuffle=True)
        
        # Chia lần 2 phần còn lại để lấy Test và Val
        # Tỉ lệ mới của test trong phần remaining
        new_test_size = test_size / remaining_size 
        test_df, val_df = train_test_split(remaining_df, test_size=(1 - new_test_size), random_state=42, shuffle=True)
    else:
        # Nếu z = 0, chỉ chia thành Train và Test
        train_df, test_df = train_test_split(df, test_size=test_size, random_state=42, shuffle=True)

    # 5. Xuất ra các file CSV tương ứng
    output_dir = os.path.dirname(args.path_to_csv_file)
    
    train_path = os.path.join(output_dir, 'train.csv')
    train_df.to_csv(train_path, index=False)
    print(f" Đã lưu: {train_path} ({len(train_df)} dòng)")
    
    test_path = os.path.join(output_dir, 'test.csv')
    test_df.to_csv(test_path, index=False)
    print(f" Đã lưu: {test_path} ({len(test_df)} dòng)")
    
    if val_df is not None:
        val_path = os.path.join(output_dir, 'vali.csv')
        val_df.to_csv(val_path, index=False)
        print(f" Đã lưu: {val_path} ({len(val_df)} dòng)")
        
    print("\n Hoàn thành xử lý dữ liệu thành công!")

if __name__ == '__main__':
    # Cấu hình bộ phân tích tham số CLI (argparse)
    parser = argparse.ArgumentParser(description="cldata - Công cụ xử lý và chia dữ liệu CSV từ dòng lệnh.")
    
    # Các options tùy chọn (flags)
    parser.add_argument('-sc', '--scale', action='store_true', help='Chuẩn hoá dữ liệu số bằng Z-score (mặc định: Không)')
    parser.add_argument('-0en', '--no-encoding', dest='encoding', action='store_false', help='Không thực hiện encoding dữ liệu chữ (mặc định: Có)')
    parser.set_defaults(encoding=True) # Thiết lập mặc định cho encoding là True
    
    # Nhận các tham số vị trí còn lại ngẫu nhiên (gồm tỉ lệ và path)
    parser.add_argument('args', nargs='*', help='Tỉ lệ x y z (tùy chọn) và đường dẫn file CSV (bắt buộc)')

    parsed_args = parser.parse_args()
    
    # Xử lý bóc tách tham số linh hoạt do x y z có thể xuất hiện hoặc không
    raw_args = parsed_args.args
    
    if len(raw_args) == 0:
        print("Lỗi: Thiếu tham số đường dẫn file CSV!")
        sys.exit(1)
        
    # Tham số cuối cùng luôn luôn là đường dẫn file
    path_to_csv_file = raw_args[-1]
    numeric_parts = raw_args[:-1] # Phần còn lại có thể là tỉ lệ x y z
    
    # Thiết lập tỉ lệ mặc định
    x, y, z = 80.0, 20.0, 0.0
    
    if len(numeric_parts) > 0:
        try:
            # Chuyển các tham số tỉ lệ sang số thực float
            ratios = [float(num) for num in numeric_parts]
            
            if len(ratios) == 2:
                x, y = ratios[0], ratios[1]
                z = 0.0
            elif len(ratios) == 3:
                x, y, z = ratios[0], ratios[1], ratios[2]
            else:
                print("Lỗi: Bạn chỉ được nhập tối đa 2 hoặc 3 tham số tỉ lệ (Ví dụ: 70 30 hoặc 70 15 15) hoặc không nhập để dùng mặc định!")
                sys.exit(1)
        except ValueError:
            print("Lỗi: Các tham số tỉ lệ train test validation phải là số thực!")
            sys.exit(1)
            
    # Gộp lại vào object args để xử lý tiếp
    parsed_args.path_to_csv_file = path_to_csv_file
    parsed_args.ratios = (x, y, z)
    
    # Chạy chương trình chính
    clean_and_split_data(parsed_args)