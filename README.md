# GPT-8
1. Для запуска проекта необходимо в файле main.py в функции main() поменять TOKEN на токен вашего бота
2. Написать команду:
```
python main.py
```
Задачу, поставленную на кейсы, мы решали таким образом:
- Предобработали датасет, используя tf-idf
- Далее применили logreg для нахождения категории вопроса
- К ответу модели применяли OHE, после чего подавали результат в BERT и получали ответ.
