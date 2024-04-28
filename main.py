from telegram import Bot
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackContext
from inference import inference
import pandas as pd

def start(update, context: CallbackContext):
    update.message.reply_text('Привет! Я помощник GeekBrains, Алексей. Ты можешь задать здесь свои вопросы')


def init_answers():
    return pd.read_csv('answer_class.csv')

def answer(update, context: CallbackContext):
    update.message.reply_text(update.message.text)


def main():
    answers = init_answers()
    predicted_class = inference('Ты пойдёшь сейчас?')
    print(answers)
    print(list(answers.loc[answers['answer_class'] == predicted_class, 'Answer'])[0])
    print('finished', predicted_class)
    TOKEN = '6812690766:AAHwAJymMGdkuUDfMFQuaX-rN9PVkBiGby4'
    updater = Updater(TOKEN, use_context=True)

    dp = updater.dispatcher
    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(MessageHandler(Filters.text & ~Filters.command, answer))
    updater.start_polling()

    updater.idle()


if __name__ == '__main__':
    main()
