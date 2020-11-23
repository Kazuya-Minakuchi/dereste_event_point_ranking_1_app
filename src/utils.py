import pickle
from inputs import input_data

# メソッドの選択に使う
def select_method(select_dict):
    """
    select_dictの中身
    key:    数字(文字列)
        name:   表示する文字
        method: 実行するメソッド
    """
    while True:
        print('モードを選んでください')
        for key, value in select_dict.items():
            print(key, ':', value['name'])
        mode_num = input_data('数字')
        # Noneが帰ってきたらキャンセル
        if mode_num is None:
            return
        print(select_dict[mode_num]['name'])
        try:
            select_dict[mode_num]['method']()
        except KeyError:
            print('正しい数字を入力してください')
        print('')

# pickleファイルがあったら開く、なかったらメッセージ表示してNoneを返す
def load_pickle(path, fault_message):
    try:
        with open(path, mode="rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        print(fault_message)
        return
