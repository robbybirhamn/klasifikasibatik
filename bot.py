"""
Simple Bot to reply to Telegram messages taken from the python-telegram-bot examples.

Source: https://github.com/python-telegram-bot/python-telegram-bot/blob/master/examples/echobot2.py
"""
import os
PORT = int(os.environ.get('PORT', 5000))
import logging
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters

from telegram.ext import Updater, Filters,CommandHandler,MessageHandler
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

model = tf.keras.models.load_model('content/saved_model_15/my_model_15label')

class_names = ['Batik Aceh', 'Batik Bali', 'Batik Bali (Poleng)', 'Batik Betawi', 'Batik Cirebon', 'Batik Kab. Kulon Progo (Geblek Renteng)', 'Batik Kalimantan Tengah', 'Batik Lasem', 'Batik Madura', 'Batik Papua', 'Batik Pati', 'Batik Pontianak', 'Batik Solo (Parang)', 'Batik Yogyakarta (Sekar Jagad)', 'Batik Yogyakarta (Tambal)']


# Enable logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO)

logger = logging.getLogger(__name__)
TOKEN = '5581120904:AAHsX04O_1vJDzZNJks9wb13YLSHYMV3-ro'

# Define a few command handlers. These usually take the two arguments update and
# context. Error handlers also receive the raised TelegramError object in error.
def start(update, context):
    """Send a message when the command /start is issued."""
    update.message.reply_text('Hi!')

def help(update, context):
    """Send a message when the command /help is issued."""
    update.message.reply_text('Help!')

def echo(update, context):
    """Echo the user message."""
    update.message.reply_text(update.message.text)

def error(update, context):
    """Log Errors caused by Updates."""
    logger.warning('Update "%s" caused error "%s"', update, context.error)


def predict_new(model, img):
    img = tf.keras.preprocessing.image.load_img(img, target_size=(224,224,3))
    x = tf.keras.preprocessing.image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    images = np.vstack([x])
    plt.imshow(img)
    img = np.expand_dims(img, axis = 0)
    predictions = model.predict(img)
    
    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = round(100 * (np.max(predictions[0])), 2)

    return predicted_class, confidence

def image(updater,context):
    photo = updater.message.photo[-1].get_file()
    photo.download("image.jpg")
    
    img = "image.jpg"
    predicted_class, confidence = predict_new(model, img)
    
    print(f"Predicted: {predicted_class}.\n Confidence: {confidence}%")
    updater.message.reply_text(f"Predicted: {predicted_class}.\n Confidence: {confidence}%")

def main():
    """Start the bot."""
    # Create the Updater and pass it your bot's token.
    # Make sure to set use_context=True to use the new context based callbacks
    # Post version 12 this will no longer be necessary
    updater = Updater(TOKEN, use_context=True)

    # Get the dispatcher to register handlers
    dp = updater.dispatcher

    # on different commands - answer in Telegram
    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(CommandHandler("help", help))

    # on noncommand i.e message - echo the message on Telegram
    dp.add_handler(MessageHandler(Filters.text, echo))
    dp.add_handler(MessageHandler(Filters.photo,image))

    # log all errors
    dp.add_error_handler(error)

    # Start the Bot
    updater.start_polling()
    # updater.start_webhook(listen="0.0.0.0",
    #                       port=int(PORT),
    #                       url_path=TOKEN)
    # updater.bot.setWebhook('https://klasifikasibatikunisbank.herokuapp.com/' + TOKEN)

    # Run the bot until you press Ctrl-C or the process receives SIGINT,
    # SIGTERM or SIGABRT. This should be used most of the time, since
    # start_polling() is non-blocking and will stop the bot gracefully.
    updater.idle()

if __name__ == '__main__':
    main()