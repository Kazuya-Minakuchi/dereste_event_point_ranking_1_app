import datetime

# データ全般のインプットに使う
def input_data(print_str):
    quit_str = 'q'
    data = input(print_str+ 'を入力してください。(' + quit_str + 'で戻る)')
    # 終了キーのときはNoneを返す
    if data == quit_str:
        print('戻ります')
        return None
    return data

# 文字列のインプットに使う
def input_str():
    input_str = input_data('文字列')
    return input_str

# 日付のインプットに使う
def input_date():
    return loop_input(convert_input_date)

# 自然数のインプットに使う
def input_natural_number():
    return loop_input(convert_input_natural_number)

# 正の数のインプットに使う
def input_positive_number():
    return loop_input(convert_input_positive_number)

# インプットを日付に変換
def convert_input_date():
    date_str = input_data('日付(yyyy-mm-dd形式)')
    # Noneが帰ってきたらキャンセル
    if date_str is None:
        return
    # 入力形式が合っていれば日付型で返す
    if is_date(date_str):
        dttm = datetime.datetime.strptime(date_str, '%Y-%m-%d')
        date = datetime.date(dttm.year, dttm.month, dttm.day)
        return date
    # 入力形式が合わないとき
    return False

# インプットを自然数に変換
def convert_input_natural_number():
    num_str = input_data('正の整数')
    # Noneが帰ってきたらキャンセル
    if num_str is None:
        return
    # 入力形式が合っていればintで返す
    if is_natural_number(num_str):
        num = int(num_str)
        return num
    # 入力形式が合わないとき
    return False

# インプットを正の数に変換
def convert_input_positive_number():
    num_str = input_data('正の数')
    # Noneが帰ってきたらキャンセル
    if num_str is None:
        return
    # 入力形式が合っていればfloatで返す
    if is_positive_number(num_str):
        num = float(num_str)
        return num
    # 入力形式が合わないとき
    return False

# 入力形式が合わない場合のループ処理
def loop_input(input_func):
    while True:
        result = input_func()
        # Noneが帰ってきたらキャンセル
        if result is None:
            return None
        # Falseが帰ってきたらもう一回
        elif result == False:
            print('入力形式が誤っています')
            continue
        return result

# yes, noのインプットに使う
def input_yes_no():
    while True:
        ym = input_data('[Y]es, [N]o')
        # Noneが帰ってきたらキャンセル
        if ym is None:
            return
        elif ym == 'Y':
            return True
        elif ym == 'N':
            return False
        print('入力値が誤っています')

# インプットから辞書型データつくる
def input_dict(input_list):
    """
    input_listは↓のdictのリスト
    name:     データ名(戻るdictのkeyになる)
    function: データ入力に使う関数
    """
    # カラムごとに取得し、辞書型で返す
    data_dict = {}
    for input_data in input_list:
        print(input_data['name'], '入力')
        data = input_data['function']()
        # Noneが帰ってきたらキャンセル
        if data is None:
            return
        data_dict[input_data['name']] = data
    return data_dict

# 文字列が日付形式か
def is_date(check_str):
    try:
        temp = datetime.datetime.strptime(check_str, '%Y-%m-%d')
        return True
    except ValueError:
        return False

# 文字列が自然数か
def is_natural_number(check_str):
    # 整数か
    try:
        num = int(check_str)
    except ValueError:
        return False
    # 正の数か
    if num > 0:
        return True
    return False

# 文字列が正の数か
def is_positive_number(check_str):
    # 数値に直せるか
    try:
        num = float(check_str)
    except ValueError:
        return False
    # 正の数か
    if num > 0:
        return True
    return False
