soundit
=======

A small library for audio manipulation.

.. code-block:: python

    import soundit as s
    s.play_output_chunks(s.chunked(s.exact(1, s.sine(440))))

Installation
------------

Simply run:

.. code-block:: bash

    $ pip install soundit[sounddevice]

The ``[sounddevice]`` extra installs sounddevice_ for playing audio through
speakers.

.. _sounddevice: https://python-sounddevice.readthedocs.io/en/0.4.4/

Usage
-----

A longer example:

.. code-block:: python

    import itertools
    import soundit as s

    indices = s.make_indices_dict()
    frequencies = s.make_frequencies_dict()
    notes = "a3 c e a g e c e d e d c a3 c a3 g3".split()

    s.play_output_chunks(s.chunked(itertools.chain.from_iterable(
        s.exact(0.5, s.sine(frequencies[indices[note]]))
        for note in notes
    )))

An even longer example:

.. code-block:: python

    import soundit as s
    s.init_piano()  # See its documentation for details on its setup

    names = "do di re ri mi fa fi so si la li ti".split()
    indices = s.make_indices_dict(names)
    music = '''
        . mi mi mi
        fa do . do
        . so mi do
        re mi,re - mi
    '''

    s.play_output_chunks(s.chunked(
        s.volume(2, s._notes_to_sound(
            s.music_to_notes(music, line_length=1.15),
            lambda name, length: s.piano(indices[name] + 1),
        ))
    ))

Links
-----

- Source Code: https://github.com/GeeTransit/soundit
- Documentation: https://geetransit.github.io/soundit/
- PyPI Releases: https://pypi.org/project/soundit/
