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


def get_notes():
    path="C:/Users/rheap/Documents/CMU/Spring 2020/AI and Culture/Project/testing_data"
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
    
    return notes

def sequence_input(notes,n_vocab):
    #sort every pitch in notes
    pitchnames=sorted(set(item for item in notes))
    #store them into dictionary
    note_to_int=dict((note,number) for number,note in enumerate(pitchnames))
    network_input=[]
    #can expriment with different sequence length 
    seq_length=100
    #creating input sequence and corresponding output sequence
    for i in range(0,len(notes)-seq_length,1):
        sequence_in=notes[i:i+seq_length]
        network_input.append([note_to_int[char] for char in sequence_in])

    num_patterns=len(network_input)
    #reshape the element based on your LSTM network
    network_input=np.reshape(network_input,(num_patterns,seq_length,1))
    #normalize input
    network_input=network_input/float(n_vocab)
    
    return network_input

def trained_network_and_weights(network_input,n_vocab):
   model=Sequential()
   model.add(LSTM(128,input_shape=(network_input.shape[1],network_input.shape[2]),return_sequences=True))
   model.add(Dropout(0.2))
   model.add(LSTM(128,return_sequences=True))
   model.add(Flatten())
   model.add(Dense(256))
   model.add(Dropout(0.3))
   model.add(Dense(n_vocab))
   model.add(Activation('softmax'))
   model.compile(loss='categorical_crossentropy',optimizer='rmsprop')


    #load the weight 
   model.load_weights('C:/Users/rheap/Documents/CMU/Spring 2020/AI and Culture/Project/train/weight.best.hdf5')
   return model 

def generate_notes(network_input,notes,n_vocab,model,pitchnames):
    start=np.random.randint(0,len(network_input)-1)
    
    #mapping integer to notes
    int_to_note=dict((number,note) for number,note in enumerate(pitchnames))
    #selecting a random note
    pattern=network_input[start]
    #print(len(pattern))
    prediction_output=[]

    #generate 500 notes
    for note in range(500):
        #print(pattern.shape)
        prediction_input=np.reshape(pattern,(1,len(pattern),1))
        prediction_input=prediction_input/float(n_vocab)

        prediction=model.predict(prediction_input,verbose=0)
        #print("done")
        #print("prediction shape:",prediction.shape)
        #print("prediction_input: ",prediction_input.shape)
        #print("pattern: ",pattern.shape)
        index=np.argmax(prediction)
        #print("index: ",index)
        result=int_to_note[index]
        
        prediction_output.append(result)
        #print(prediction_output)

        np.append(pattern,index)
        #print(pattern.shape)
        pattern=pattern[0:len(pattern)]
        #print(pattern.shape)
    
    return prediction_output

def generate_midiFile(prediction_output):
    output_notes=[]
    offset=0

    for pattern in prediction_output:
        #pattern = pattern.split()
        #temp = pattern[0]
        #duration = pattern[1]
        #pattern = temp
        # pattern is a chord
        if ('.' in pattern) or pattern.isdigit():
            notes_in_chord = pattern.split('.')
            notes = []
            for current_note in notes_in_chord:
                new_note = music21.note.Note(int(current_note))
                new_note.storedInstrument = music21.instrument.Piano()
                notes.append(new_note)
            new_chord = music21.chord.Chord(notes)
            new_chord.offset = offset
            output_notes.append(new_chord)

        # pattern is a note
        else:
            new_note = music21.note.Note(pattern)
            new_note.offset = offset
            new_note.storedInstrument = music21.instrument.Piano()
            output_notes.append(new_note)
        # increase offset each iteration so that notes do not stack
        offset += 0.5

    midi_stream = music21.stream.Stream(output_notes)

    midi_stream.write('midi', fp='C:/Users/rheap/Documents/CMU/Spring 2020/AI and Culture/Project/train/test_output.mid')
 
def convert_to_float(frac_str):
    try:
        return float(frac_str)
    except ValueError:
        num, denom = frac_str.split('/')
        try:
            leading, num = num.split(' ')
            whole = float(leading)
        except ValueError:
            whole = 0
        frac = float(num) / float(denom)
        return whole - frac if whole < 0 else whole + frac

def predict():
    #finding notes for testing data 
    notes=pickle.load(open("C:/Users/rheap/Documents/CMU/Spring 2020/AI and Culture/Project/train/notes.pickle","rb"))
    #check 1 
    print("step 1: notes generated")
    #print(notes)
    n_vocab=len(set(notes))
    pitchnames=sorted(set(item for item in notes))
    #getting input for all the notes 
    network_input=sequence_input(notes,n_vocab)
    #check 2 
    print("step 2: network input is done")
    #getting a trained network 
    model=trained_network_and_weights(network_input,n_vocab)
    #check 3 
    print("step 3: trained model created")
    #predicting notes 
    prediction_output=generate_notes(network_input,notes,n_vocab,model,pitchnames)
    #check 4
    print("step 4: notes generated")
    #generating a midi file 
    generate_midiFile(prediction_output)
    #check 5
    print("step 5: midi file created")


predict()
print("finally done")


