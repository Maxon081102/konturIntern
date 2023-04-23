from pipeline import Pipeline

def get_context_from_file(file_name):
    try:
        with open(file_name, 'r') as file:
            data = file.read()
        return data
    except Exception as e:
        return None

def run_programm(pipeline):
    print("Введите файл с контекстом (формат файла txt)")
    file_name = input()
    context = get_context_from_file(file_name)
    if context is None:
        print("Файл не существует")
        return
    
    print("Что Вы хотите ?")
    print("  - \"обеспечение исполнения контракта\" введите 0")
    print("  - \"обеспечение гарантийных обязательств\" введите 1")
    if int(input()) == 0:
        task = "обеспечение исполнения контракта"
    else:
        task = "обеспечение гарантийных обязательств"
    print()
    answer = pipeline(context, task)
    print(answer)
    print("-----------------------------------------------------------------------------------")
    print()
    
if __name__ == "__main__":
    print("Модель загружается . . .")
    pipeline = Pipeline()
    print("Модель загрузилась")
    while(True):
        run_programm(pipeline)