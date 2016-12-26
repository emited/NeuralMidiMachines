from music21 import *
import numpy as np
import glob
from collections import OrderedDict
import pandas as pd


def parseMidiFile(fn, keep_rythm=False):

    '''Loads a midi file, using music21,
    flattens out the different tracks 
    and saves the results in OrderedDict,
    using the offset (time) as key and the
    values : set(pitch_of_note_1, ...)
     '''

    #load and convert to music21 format
    midi_data = converter.parse(fn)
    
    if not keep_rythm:
        unique_offsets = np.unique([n.offset for n in midi_data.flat.notes])
        offset2discr = {offset:i for i, offset in enumerate(unique_offsets)}
        
    seq = OrderedDict()
    for n in midi_data.flat.notes:
        
        if keep_rythm:
            offset = n.offset
            
        else:
            offset = offset2discr[n.offset]

        seq.setdefault(offset, set())
        if isinstance(n, note.Note):
            seq[offset].add(n.pitch.midi)
        
        #if n is a chord, transform to seq
        elif isinstance(n, chord.Chord):
            for chord_note in n:
                seq[offset].add(chord_note.pitch.midi)

    return seq.items()


def writeToMidiFile(fn, seq):
    recons = stream.Part()
    recons.insert(0, instrument.ElectricGuitar())
    i = 0
    for offset, pitches in seq:
        i += 1
        if len(pitches) > 1:
            recons.insert(offset, chord.Chord(pitches))
        else:
            recons.insert(offset, note.Note(list(pitches)[0]))
    
    recons.write('midi', fn)


def parseSeqFile(fn):
    seq = []
    with open(fn, 'r') as f:
        for row in f:
            ns = row.split(',')[1:-1]
            offset = int(float(row.split(',')[0]))
            nns = [int(float(n)) for n in ns if len(n)>0]
            seq.append((offset, nns))
    return seq


def writeSeqFile(fn, seq):
    df = pd.DataFrame(dict(seq).values())
    df.to_csv(fn, header=False)



def main():

    print 'Converting midi files to seq files....'

    midi_fns = glob.glob('midi/*')
    seqs = []
    error_count = 0
    for i, midi_fn in enumerate(midi_fns):
        
        try:
            print 'Preprocessing file '+str(i)+' of '+str(len(midi_fns))\
                +' '+midi_fn+'...'

            seq = parseMidiFile(midi_fn, keep_rythm=False)

            seq_fn = 'seqs/' + midi_fn[5:-4] + '.seq'
            
            writeSeqFile(seq_fn, seq)

            seqs.append(seq)

        except:
            error_count += 1
            print 'Error reading '+midi_fn+' !'

    print 'Preprocessed '+str(i)+' files with '+str(error_count)+' errors!'

    print 'Building batches...'
    overlap = 25
    seq_length = 50

    batches = []
    for seq in seqs:
        for i in range(0, len(seq) - seq_length + 1, overlap):
            batches.append(seq[i:i + seq_length])
    
    print 'Created '+str(len(batches))+' batches.'

    print 'Saving batches to hdf5...'


    print 'Done !'



if __name__ == "__main__":
    main()