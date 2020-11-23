import pickle
from inputs import input_data

# メソッドの選択に使う
def select_method(selections):
    """
    selectionsの中身
    key:    数字(文字列)
        name:   表示する文字
        method: 実行するメソッド
    """
    while True:
        # 選択肢表示
        print('モードを選んでください')
        for key, value in selections.items():
            print(key, ':', value['name'])
        # 選択
        mode_num = input_data('数字')
        # Noneが帰ってきたらキャンセル
        if mode_num is None:
            return
        print(selections[mode_num]['name'])
        # 正しい値が入力されていたら、実行
        try:
            selections[mode_num]['method']()
        # 値が間違っていたら、やり直し
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
