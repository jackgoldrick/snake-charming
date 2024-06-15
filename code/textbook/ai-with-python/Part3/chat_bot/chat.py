import speech_recognition as sr
import sys
import numpy as np
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras import layers, activations, models, Model, preprocessing # type: ignore
import os
import yaml
from tensorflow.keras import preprocessing , utils # type: ignore

import re

def create_and_train(dir_path='data/set', tokenizer=Tokenizer()):
    """ Create the Encoder Model
    """
    questions, answers = load_data(dir_path)
    encoder_input_data, decoder_input_data, decoder_output_data, maxlen_questions, maxlen_answers, VOCAB_SIZE, tokenizer = chef_data(questions, answers, tokenizer)
    
    encoder_inputs = tf.keras.layers.Input(shape=( maxlen_questions , ))
    encoder_embedding = tf.keras.layers.Embedding( VOCAB_SIZE, 200, mask_zero=True ) (encoder_inputs)
    encoder_outputs , state_h , state_c = tf.keras.layers.LSTM(200 , return_state=True )( encoder_embedding )
    encoder_states = [ state_h , state_c ]
    
    decoder_inputs = tf.keras.layers.Input(shape=( maxlen_answers , ))
    decoder_embedding = tf.keras.layers.Embedding( VOCAB_SIZE, 200, mask_zero=True) (decoder_inputs)
    decoder_lstm = tf.keras.layers.LSTM( 200 , return_state=True , return_sequences=True, name="LSTM")
    decoder_outputs , _ , _ = decoder_lstm ( decoder_embedding , initial_state=encoder_states )
    decoder_dense = tf.keras.layers.Dense( VOCAB_SIZE , activation=tf.keras.activations.softmax, name="dense_decoder" )
    
    output = decoder_dense ( decoder_outputs )
    model = tf.keras.models.Model([encoder_inputs, decoder_inputs], output )
    model.compile(optimizer=tf.keras.optimizers.RMSprop(), loss='categorical_crossentropy') # type: ignore
    model.summary()
    
    model.fit([encoder_input_data , decoder_input_data], decoder_output_data, batch_size=50, epochs=150, verbose=0 )
    # model.save( 'models/model.h5' )
    
    tokenizer_dict = { tokenizer.word_index[i]:i for i in tokenizer.word_index}
    print(tokenizer_dict)
    
    output = model.predict([encoder_input_data[0,np.newaxis], decoder_input_data[0,np.newaxis]])
    
    sampled_word_indexes = np.argmax(output[0],1)
    sentence = ""
    maxlen_answers = 74
    
    for sampled_word_index in sampled_word_indexes:
        sampled_word = None
        # Check if sampled_word_index exists in tokenizer_dict
        if sampled_word_index in tokenizer_dict:
            sampled_word = tokenizer_dict[sampled_word_index]
        sentence += ' {}'.format(sampled_word)
        
        if sampled_word == 'end' or len(sentence.split()) > maxlen_answers:
            break
    print(sentence)
        
    enc_model, dec_model = make_inference_model(model, encoder_inputs, decoder_inputs, encoder_states, encoder_embedding)
    return enc_model, dec_model, maxlen_questions, maxlen_answers
    



def make_inference_model(model, encoder_inputs, decoder_inputs,  encoder_states, decoder_embedding):
    decoder_lstm = model.get_layer('LSTM')
    decoder_dense = model.get_layer('dense_decoder')
    encoder_model = tf.keras.models.Model(encoder_inputs, encoder_states)
  
  
  
    # decoder_state_input_h = tf.keras.layers.Input(shape=( 200 ,))
    # decoder_state_input_c = tf.keras.layers.Input(shape=( 200 ,))
    # decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    
    # Assuming the rest of `make_inference_model` is correctly set up, focusing on decoder model setup
    decoder_state_input_h = tf.keras.layers.Input(shape=(200,), name='input_h')
    decoder_state_input_c = tf.keras.layers.Input(shape=(200,), name='input_c')
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

    decoder_outputs, state_h, state_c = decoder_lstm(decoder_embedding, initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]

    # Output of the dense layer
    decoder_outputs = decoder_dense(decoder_outputs)

    # Define the decoder model
    decoder_model = tf.keras.models.Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)
        
        
    # decoder_outputs, state_h, state_c = decoder_lstm(decoder_embedding , initial_state=decoder_states_inputs)
    # decoder_states = [state_h, state_c]
    
    
    # decoder_outputs = decoder_dense(decoder_outputs)
    # decoder_model = tf.keras.models.Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)
    
    return encoder_model , decoder_model
    


def load_data(dir_path='data/set/'):
    
    #dir_path = 'data/'
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
        

    return  questions, answers

def chef_data(questions, answers, tokenizer = preprocessing.text.Tokenizer()):
    
    
    vocab = []
        
    
    tokenizer.fit_on_texts( questions + answers )
    VOCAB_SIZE = len( tokenizer.word_index )+1
    print('VOCAB SIZE : {}'.format( VOCAB_SIZE ))
    
    for word in tokenizer.word_index:
        vocab.append(word)
        
    def tokenize(sentences):
        tokens_list = []
        vocabulary = []
        for sentence in sentences:
            sentence = sentence.lower()
            sentence = re.sub('[^a-zA-Z]', ' ', sentence)
            tokens = sentence.split()
            vocabulary += tokens
            tokens_list.append(tokens)
        return tokens_list, vocabulary
    
    # p = tokenize( questions + answers )
    # model = Word2Vec( p[ 0 ] )
    # embedding_matrix = np.zeros( ( VOCAB_SIZE , 100 ) )
    # for i in range( len( tokenizer.word_index ) ):
    #     embedding_matrix[ i ] = model[ vocab[i] ]

    # encoder_input_data
    tokenized_questions = tokenizer.texts_to_sequences(questions)
    maxlen_questions = max( [ len(x) for x in tokenized_questions] ) 
    padded_questions = preprocessing.sequence.pad_sequences(tokenized_questions , maxlen=maxlen_questions , padding='post' )
    encoder_input_data = np.array( padded_questions )
    print( encoder_input_data.shape , maxlen_questions )



    # decoder_input_data
    tokenized_answers = tokenizer.texts_to_sequences(answers)
    maxlen_answers = max([len(x) for x in tokenized_answers])
    padded_answers = preprocessing.sequence.pad_sequences(tokenized_answers , maxlen=maxlen_answers , padding='post')
    decoder_input_data = np.array(padded_answers)
    print(decoder_input_data.shape , maxlen_answers)

    # decoder_output_data
    tokenized_answers = tokenizer.texts_to_sequences(answers)
    for i in range(len(tokenized_answers)):
        tokenized_answers[i] = tokenized_answers[i][1:]
        
    padded_answers = preprocessing.sequence.pad_sequences(tokenized_answers , maxlen=maxlen_answers , padding='post' )
    onehot_answers = utils.to_categorical( padded_answers , VOCAB_SIZE )
    decoder_output_data = np.array( onehot_answers )
    print( decoder_output_data.shape )

    return encoder_input_data, decoder_input_data, decoder_output_data, maxlen_questions, maxlen_answers, VOCAB_SIZE, tokenizer
    
           


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
            print("Processing the audio...")
            words = r.recognize_google(audio) # recognize speech using Google Speech Recognition
            print("You said " + words) #recognize speech using Google Speech Recognition
            return str(words)
        except LookupError: # speech is unintelligible
            print("Could not understand audio")
            
        except:
            print("Please retry...")
        
        
        
        