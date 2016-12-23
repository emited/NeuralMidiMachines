from music21 import *
import numpy as np
import glob
from collections import OrderedDict
import pandas as pd


def parseMidiFile(data_fn, keep_rythm=False):
    '''Loads a midi file, using music21,
    flattens out the different tracks 
    and saves the results in OrderedDict,
    using the offset (time) as key and the
    values : set(pitch_of_note_1, ...)
     '''

    #load and convert to music21 format
    midi_data = converter.parse(data_fn)
    
    #I guess I can only take first one???
    part = midi_data[0]

    #guetting all Voice streams
    streams = part.getElementsByClass(stream.Voice)
    
    #flattening out the parts the streams
    flat = streams.flat
    if not keep_rythm: offset = -1
    notes = OrderedDict()
    for n in flat.notes:
        
        if keep_rythm:
            offset = n.offset
        else:
            offset += 1

        notes.setdefault(offset, set())

        if isinstance(n, note.Note):
            notes[offset].add(n.pitch.midi)
        
        #if n is a chord, transform to notes
        elif isinstance(n, chord.Chord):
            for chord_note in n:
                notes[offset].add(chord_note.pitch.midi)

    return notes.items()



def reconstructMidiFile(data, fn):
    
    recons = stream.Part()
    
    recons.insert(0, instrument.ElectricGuitar())
    i = 0
    for offset, pitches in notes.iteritems():
        i += 1
        if len(pitches) > 1:
            recons.insert(offset, chord.Chord(pitches))
        else:
            recons.insert(offset, note.Note(list(pitches)[0]))

    print('writing reconstruction to '+fn+'...')
    recons.write('midi', fn)

    return recons



def main():

    midi_fns = glob.glob('midi/*')
    
    for midi_fn in midi_fns:
        
        print 'preprocessing '+midi_fn+'...'
        
        try:
            #parse and convert midi files
            notes = parseMidiFile(midi_fn, keep_rythm=False)

            #saving to csvs
            # /!\ only when keep_rythm is False
            df = pd.DataFrame(dict(notes).values())
            csv_fn = midi_fn[5:-4]+'.csv'
            df.to_csv('csv/'+csv_fn, header=False)
        
        except:
            print 'error with'+midi_fn
if __name__ == "__main__":
    main()