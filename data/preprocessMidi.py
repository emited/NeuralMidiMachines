from music21 import *
import numpy as np
import glob
from collections import OrderedDict
import torchfile
import cPickle as pkl

def parseMidiFile(fn, keep_rythm=False, transpose_to=None):

    '''Loads a midi file, using music21,
    flattens out the different tracks 
    and saves the results in OrderedDict,
    using the offset (time) as key and the
    values : set(pitch_of_note_1, ...)
     '''

    #load and convert to music21 format
    midi_data = converter.parse(fn)
    
    #if we do not use rythm, we set spacing between each
    #note / chord to 1.
    if not keep_rythm:
        unique_offsets = np.unique([n.offset for n in midi_data.flat.notes])
        offset2discr = {offset:i for i, offset in enumerate(unique_offsets)}

    if transpose_to:
        k = midi_data.analyze('key')
        i = interval.Interval(k.tonic, pitch.Pitch(transpose_to))
        midi_data = midi_data.transpose(i)

    if not keep_rythm:
        seq = OrderedDict()
        for n in midi_data.flat.notes:
            
            offset = offset2discr[n.offset]

            seq.setdefault(offset, set())
            if isinstance(n, note.Note):
                seq[offset].add(n.pitch.midi)
            
            #if n is a chord, get each note of n
            elif isinstance(n, chord.Chord):
                for chord_note in n:
                    seq[offset].add(chord_note.pitch.midi)

        return seq.items()
    
    else:
        
        seq = []
        last_offset = midi_data.flat.notes[0].offset
        for n in midi_data.flat.notes:
            offset = '%.5f'%(float(n.offset) - float(last_offset))
            last_offset = n.offset
            if isinstance(n, note.Note):
                seq.append((offset, {n.pitch.midi}))
            elif isinstance(n, chord.Chord):
                for chord_note in n:
                    seq.append((offset, {chord_note.pitch.midi}))


        return seq


def writeMidiFile(fn, seq):
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


def writeMidiFileFromTensor(tensor_fn, out_fn, method='sample'):
    
    tensor = np.squeeze(torchfile.load(tensor_fn))
    
    recons = stream.Part()
    recons.insert(0, instrument.ElectricGuitar())

    for offset, notes in enumerate(tensor):
        
        if method == 'all':
            for pitch, volume in enumerate(notes):
                if volume > 0:
                    n = note.Note(pitch)
                    n.volume = volume * 100
                    recons.insert(offset, n)

        elif method == 'sample':
             s = np.random.choice(len(notes), p=notes/notes.sum())
             n = note.Note(s)
             recons.insert(offset, n)

        elif method == 'argmax':
            n = note.Note(notes.argmax())
            recons.insert(offset, n)

    print(recons.show('text'))
    recons.write('midi', out_fn)


def parseSeqFile(file):
    seq = []
    for line in file:
        offset, notes = line.split(':')
        notes = notes.replace('\n', '').split(',')
        notes = [int(n) for n in notes]
        try:
            seq.append((float(offset), notes))
        except:
            print(offset)
    return seq


def writeSeqFile(file, seq):
    for offset, notes in seq:
        line = str(offset)+':'+str(list(notes))[1:-1]+'\n'
        file.write(line)


def main():

    
    midi_fns = glob.glob('midi/*')
    path = 'seqs_rythm/'
    transpose_to = 'C'

    print 'Converting midi files to seq files....'

    error_count = 0
    for i, midi_fn in enumerate(midi_fns):
        
        try:
            print 'Preprocessing file '+str(i+1)+' of '+str(len(midi_fns))\
                +' '+midi_fn+'...'

            seq = parseMidiFile(midi_fn, keep_rythm=True, transpose_to=transpose_to)

            seq_fn = path + '/' + midi_fn[5:-4] + '.seq'
            
            file = open(seq_fn, 'w')
            writeSeqFile(file, seq)
            file.close()

        except:
            error_count += 1
            print 'Error reading '+midi_fn+' !'

    print 'Preprocessed '+str(i+1)+' files with '+str(error_count)+' errors!'
    
    
    #tensor_fn = '../samples/sample_590.t7'
    #out_fn = 'out.mid'
    #writeMidiFileFromTensor(tensor_fn, out_fn, method='argmax')

    #seqs = []
    #seq_fns = glob.glob('seqs_transposed/*')
    #for seq_fn in seq_fns:
    #    seqs.append(parseSeqFile(seq_fn))
    #print('saving seqs...')
    #pkl.dump(seqs, open('all_seqs_transposed.pkl', 'w'))
    #
    #overlap = 25
    #seq_length = 50
    #
    #batch_file = open('batches.seqs', 'w')
    #
    #batches = []
    #for seq, seq_fn in zip(seqs, seq_fns):
    #    for i in range(0, len(seq) - seq_length + 1, overlap):
    #        batch = seq[i:i + seq_length]
    #        batch_file.write(str(len(batch))+','+seq_fn+'\n')
    #        batches.append(batch)
    #        writeSeqFile(batch_file, batch)
    #
    #print 'Created '+str(len(batches))+' batches.'
    #
    #batch_file.close()
    #print 'Done !'



if __name__ == "__main__":
    main()