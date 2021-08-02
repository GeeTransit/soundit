"""Plays a song. This requires sounddevice to be installed

The song is based on Digitized by GeeTransit using Online Sequencer located at
https://onlinesequencer.net/822200.

"""
import soundit as s

# Dictionaries to convert note names into indices / frequencies
indices = s.make_indices_dict("do di re ri mi fa fi so si la li ti".split())
frequencies = s.make_frequencies_dict()

# The music to parse and play
music = '''
    # names="do di re ri mi fa fi so si la li ti".split()
    # line_length=4.36
    # offset=13
/   # offset=1

    . mi mi mi fa       do . do .           so mi do re         mi,re - mi
/   .

    la3 mi mi mi        fa do . do          . so mi do          re mi,re - mi
/   la3 la3 la3 fa3     fa3 fa3 fa3 fa3     do do do so3        so3 so3 si3 si3

    do mi mi mi         fa do . do          . so mi do          re mi,re - mi
/   la3 la3 la3 fa3     fa3 fa3 fa3 fa3     do do do so3        so3 so3 si3 si3

    do mi mi mi         fa do . do          . so mi do          re mi,re - mi
/   la3 la3 la3 la3     la3 la3 la3 la3     la3 la3 la3 la3     la3 la3 la3 la3

    do mi mi mi         fa do . do          . so mi do          . . . .
/   la3 la3 la3 la3     la3 la3 la3 la3     la3 la3 la3 la3     la3,do,mi,so la,do5,mi5,so5 la5 .

    do so mi do         re re,mi so mi      do so mi do         re re,mi so re
/   la3 la3 la3 fa3     fa3 fa3 fa3 fa3     do do do so3        so3 so3 si3 si3

    do so mi do         re re,mi so mi      do so mi do         re re,mi so re
/   la3 la3 la3 fa3     fa3 fa3 fa3 fa3     do do do so3        so3 so3 si3 si3

    - do . la3          . la3 do re         so3 do mi do        re do,re - do
/   la3 la3 la3 fa3     fa3 fa3 fa3 fa3     do do do so3        so3 so3 si3 si3

    - do . la3          . la3 do re         so3 do mi do        re do,re - do
/   la3 la3 la3 fa3     fa3 fa3 fa3 fa3     do do do so3        so3 so3 si3 si3

    do la3 - so3        do re,mi - re       . . do so3          re mi re do
/   .

    . . do so3          fa mi,re - do       - so3 re do         mi so re do
/   .

    la3 mi mi mi        fa do . do          . so mi do          re mi,re - mi
/   la3 . la3 .         fa3 . fa3 .         do . do .           so3 . so3 .

    do mi mi mi         fa do . do          . so mi do          re mi,re - mi
/   la3 . la3 .         fa3 . fa3 .         do . do .           so3 . so3 .

    do mi mi mi         fa do . do          . so mi do          re mi,re - mi
/   la3 la3 la3 fa3     fa3 fa3 fa3 fa3     do do do so3        so3 so3 si3 si3

    do mi mi mi         fa do . do          . so mi do          . . . .
/   la3 la3 la3 fa3     fa3 fa3 fa3 fa3     do do do so3        la3,do,mi,so la,do5,mi5,so5 la5 .

    do so mi do         re re,mi so mi      do so mi do         re re,mi so re
/   la3 la3 la3 fa3     fa3 fa3 fa3 fa3     do do do so3        so3 so3 si3 si3

    do so mi do         re re,mi so mi      do so mi do         re re,mi so re
/   la3 la3 la3 fa3     fa3 fa3 fa3 fa3     do do do so3        so3 so3 si3 si3

    - do . la3          . la3 do re         so3 do mi do        re do,re - do
/   la3 la3 la3 fa3     fa3 fa3 fa3 fa3     do do do so3        so3 so3 si3 si3

    - do . la3          . la3 do re         so3 do mi do        re do,re - -
/   la3 la3 la3 fa3     fa3 fa3 fa3 fa3     do do do so3        so3 so3 si3 si3

    do . . .
/   la3 . . .
'''

# Some math to turn BPM into seconds per line
BPM = 110  # beats / minute
SECONDS_PER_MINUTE = 60  # seconds / minute
BEATS_PER_BAR = 4  # beats / bar
BARS_PER_LINE = 2  # bars / line
LINE_LENGTH = (1/BPM) * SECONDS_PER_MINUTE * BEATS_PER_BAR * BARS_PER_LINE

# Splitting the top and bottom parts
top_music, bottom_music = s.split_music(music)

# Top part is a triangle wave
top_sound = s._notes_to_sound(
    s.music_to_notes(top_music, line_length=LINE_LENGTH),
    lambda name, length: s.fade(s.cut(
        length,
        (0.5*x for x in s.triangle(frequencies[indices[name] + 1 + 12])),
    )),
)

# Bottom part is a square wave
bottom_sound = s._notes_to_sound(
    s.music_to_notes(bottom_music, line_length=LINE_LENGTH),
    lambda name, length: s.fade(s.cut(
        length,
        (0.08*x for x in s.square(frequencies[indices[name] + 1])),
    )),
)

# Mix music by averaging out the top and bottom
s.play_output_chunks(s.chunked(
    0.5*x + 0.5*y
    for x, y in zip(top_sound, bottom_sound)
))
