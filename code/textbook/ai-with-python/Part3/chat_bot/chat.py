import speech_recognition as sr
import sys
import numpy as np
import tensorflow as tf
import pickle
from tensorflow.keras import layers, activations, models, preprocessing # type: ignore
import os
import yaml
from tensorflow.keras import preprocessing , utils # type: ignore

def load_data(dir_path):
    
    # dir_path = 'data/'
    files_list = os.listdir(dir_path + os.sep)
    questions = list()
    answers = list()
    for filepath in files_list:
        stream = open( dir_path + os.sep + filepath , 'rb')
        docs = yaml.safe_load(stream)
        conversations = docs['conversations']
        
        for con in conversations:
            if len( con ) > 2 :
                questions.append(con[0])
                replies = con[ 1 : ] 
                ans = ''
                for rep in replies:
                    ans += ' ' + rep
                answers.append( ans )
            elif len( con )> 1:
                questions.append(con[0])
                answers.append(con[1])
                
    answers_with_tags = list()
    for i in range( len( answers ) ):
        if type( answers[i] ) == str:
            answers_with_tags.append( answers[i] )
        else:
            questions.pop( i )
    
    answers = list()
    for i in range( len( answers_with_tags ) ) :
        answers.append( '<START> ' + answers_with_tags[i] + ' <END>' )
        
    tokenizer = preprocessing.text.Tokenizer()
    tokenizer.fit_on_texts( questions + answers )
    VOCAB_SIZE = len( tokenizer.word_index )+1
    print('VOCAB SIZE : {}'.format( VOCAB_SIZE ))




def record_audio(): 
    """ Capture audio from the microphone and return the audio data"""
    r = sr.Recognizer()
    
    while True:  # Start of the loop
        user_input = input("Type 'exit' to stop or press Enter to continue: ")
        if user_input.lower() == "exit":
            print("Exiting...")
            break  # Exit the fxn if user types 'exit
        
            
        print("please say something in 4 seconds... and wait for 4 seconds for the answer.....")
        print("Accessing Microphone..")
        
        try:
            with sr.Microphone() as source:
                r.adjust_for_ambient_noise(source, duration=4)
        # use the default microphone as the audio source, duration higher means environment noisier
                print("Waiting for you to speak...")
                audio = r.listen(source) # listen for the first phrase and extract it into audio data
        except (ModuleNotFoundError,AttributeError):
            print('Please check installation')
            sys.exit(0)
        try:

            print("You said " + r.recognize_google(audio)) #recognize speech using Google Speech Recognition
            
        except LookupError: # speech is unintelligible
            print("Could not understand audio")
            
        except:
            print("Please retry...")
        
        
        
        