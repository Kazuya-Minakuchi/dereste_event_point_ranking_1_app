from typing import Any
from dataclasses import dataclass
from inputs import input_integer

class Selections:
    def __init__(self, selection_list):
        self.selections = {i: selection for i, selection in enumerate(selection_list)}
    
    # 実行するメソッド選ぶ
    def select_method(self):
        while True:
            self.show_selection()
            # 番号選択
            mode_num = input_integer()
            # Noneが帰ってきたらキャンセル
            if mode_num is None:
                return
            self.execute_method(mode_num)
    
    # 選択肢表示
    def show_selection(self):
        print('モードを選んでください')
        for i, selection in self.selections.items():
            print(i, ':', selection.name)
    
    # 入力された番号のメソッドを実行
    def execute_method(self, num):
        # 正しい値が入力されていたら、実行
        try:
            self.selections[num].execute_method()
        # 値が間違っていたらやりなおし
        except KeyError:
            print('正しい数字を入力してください')

@dataclass
class Selection:
    name:   str
    method: Any
    
    def execute_method(self):
        self.method()

if __name__ == '__main__':
    selections = Selections([
        Selection(
                name = 'name1',
                method = (lambda: print('method1')),
        ),
        Selection(
            name = 'name2',
            method = (lambda: print('method2')),
        ),
    ])
    selections.select_method()
