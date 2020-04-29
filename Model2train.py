import music21
import pickle
import glob
import numpy as np
import keras
from keras import utils
from keras.models import Sequential
from keras.layers import Bidirectional, Dense, Dropout, Activation, LSTM, Flatten
from keras.callbacks import ModelCheckpoint
from keras_self_attention import SeqSelfAttention

#train data one has 100 elements 


def get_notes():
    path="C:/Users/rheap/Documents/CMU/Spring 2020/AI and Culture/Project/training_data/train_data_one"
    notes=[]
    for file in glob.glob(path+"/*.midi"):
        midi=music21.converter.parse(file)
        notes_to_parse=[]
        try:
            parts=music21.instrument.partitionByInstrument(midi)
        except:
            pass
        if parts:
            notes_to_parse=parts.parts[0].recurse()
        else:
            notes_to_parse=midi.flat.notes
            
        for element in notes_to_parse:
            if isinstance(element,music21.note.Note):
                notes.append(str(element.pitch)+ " " +str(element.quarterLength))
            elif isinstance(element,music21.chord.Chord):
                notes.append('.'.join(str(n) for n in element.normalOrder))
    with open('C:/Users/rheap/Documents/CMU/Spring 2020/AI and Culture/Project/train/notes.pickle','wb') as filepath:
        pickle.dump(notes,filepath)
    
    return notes

def sequence_input(notes,n_vocab):
    #sort every pitch in notes
    pitchnames=sorted(set(item for item in notes))
    #store them into dictionary
    note_to_int=dict((note,number) for number,note in enumerate(pitchnames))
    network_input=[]
    network_output=[]
    #can expriment with different sequence length 
    seq_length=100
    #creating input sequence and corresponding output sequence
    for i in range(0,len(notes)-seq_length,1):
        sequence_in=notes[i:i+seq_length]
        sequence_out=notes[i+seq_length]
        network_input.append([note_to_int[char] for char in sequence_in])
        network_output.append(note_to_int[sequence_out])

    num_patterns=len(network_input)
    #reshape the element based on your LSTM network
    network_input=np.reshape(network_input,(num_patterns,seq_length,1))
    #normalize input
    network_input=network_input/float(n_vocab)
    #one hot encoding for output
    network_output=utils.to_categorical(network_output)
    
    return (network_input,network_output)

#creating a network that is bidirectional, LSTM and then attention
def create_network(network_input,n_vocab):
   #using just 2 LSTM network
   model=Sequential()
   model.add(Bidirectional(LSTM(256,input_shape=(network_input.shape[1],network_input.shape[2]),return_sequences=True)))
   model.add(Dropout(0.3))
   model.add(SeqSelfAttention(attention_activation='sigmoid'))
   model.add(Dropout(0.3))
   model.add(Dense(128))
   model.add(Flatten())
   model.add(Dense(n_vocab))
   model.add(Activation('softmax'))
   model.compile(loss='categorical_crossentropy',optimizer='rmsprop')

   return model

def train_data(model,network_input,network_output):
    filepath="C:/Users/rheap/Documents/CMU/Spring 2020/AI and Culture/Project/train/weight.best.hdf5"
    checkpoint=ModelCheckpoint(
        filepath, monitor="loss",
        verbose=0,
        save_best_only=True,
        mode='min'
    )

    callbacks_list=[checkpoint]
    model.fit(network_input,network_output,epochs=50,batch_size=64,callbacks=callbacks_list)

def train():
    #finding all notes and chords as data 
    notes=get_notes()
    #first check 
    print("First Step: notes list is done")
    #getting distinct number of pitch 
    n_vocab=len(set(notes))
    #getting processed final input and output 
    network_input,network_output=sequence_input(notes,n_vocab)
    #putting a second check 
    print("Second Step:processed input and output matrix is formed")
    #getting created network 
    model=create_network(network_input,n_vocab)
    #putting a third check
    print("Third Step: model is created")
    #training data
    train_data(model,network_input,network_output)


train()
#final check 
print("final Check:")
print("data has been trained")