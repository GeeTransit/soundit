"""Plays a song. This requires sounddevice to be installed

The song is based on Megalovania by Toby Fox. It's kinda incomplete too.

"""
import soundit as s

# Dictionaries to convert note names into indices / frequencies
indices = s.make_indices_dict("do di re ri mi fa fi so si la li ti".split())
frequencies = s.make_frequencies_dict()

# The music to parse and play
music = '''
# names="do di re ri mi fa fi so si la li ti".split()
# offset=6
# line_length=2.2

la3 la3 la - mi - - ri - re - do - la3 do re
so3 so3 la - mi - - ri - re - do - la3 do re
fi3 fi3 la - mi - - ri - re - do - la3 do re
fa3 fa3 la - mi - - ri - re - do - la3 do re

la3 la3 la - mi - - ri - re - do - la3 do re
so3 so3 la - mi - - ri - re - do - la3 do re
fi3 fi3 la - mi - - ri - re - do - la3 do re
fa3 fa3 la - mi - - ri - re - do - la3 do re

do - do do - do - do - la3 - la3 - - - -
do - do do - re - ri - re do la3 do re - -
do - do do - re - ri - mi - so - mi - -
la - la - la mi la so - - - - - - - -

mi - mi mi - mi - mi - re - re - - - -
mi - mi mi - mi - re - mi - so - mi re -
la do mi do so do mi do re do re mi so mi re do
la3 - ti3 - do la3 do so - - - - - - - -

la3 - - - - - - - do la3 do re ri re do la3
do la3 do - re - - - - - - - - - ri mi
la - re mi re do ti3 la3 do - re - mi - so -
la - la - la mi la so - - - - - - - -

do - re - mi - do5 - ti - - - si - - -
ti - - - do5 - - - re5 - - - ti - - -
mi5 - - - - - - - mi5 re5 ti si mi do la3 si3
so3 - - - - - - - si3 - - - - - - -

mi3 - - - - - - - - - - - do - - -
ti3 - - - - - - - si3 - - - - - - -
la3
-
'''

# Some math to turn BPM into seconds per line
BPM = 120  # beats / minute
SECONDS_PER_MINUTE = 60  # seconds / minute
BEATS_PER_BAR = 4  # beats / bar
BARS_PER_LINE = 1  # bars / line
LINE_LENGTH = (1/BPM) * SECONDS_PER_MINUTE * BEATS_PER_BAR * BARS_PER_LINE

# Top part is a triangle wave
sound = s._notes_to_sound(
    s.music_to_notes(music, line_length=LINE_LENGTH),
    lambda name, length: s.fade(s.cut(
        length,
        (0.08*x for x in s.sawtooth(frequencies[indices[name] + 5])),
    )),
)

# Play the music
s.play_output_chunks(s.chunked(sound))
