// "What am I seeing?": short notes for the current mode, and things that catch people out.

export interface Note {
  heading: string;
  body: string;
}

const MODE_NOTES: Record<string, Note> = {
  am: {
    heading: "AM",
    body:
      "The sound is in the arrow's LENGTH. Tuned exactly, the arrow barely rotates and the constellation is a short " +
      "line. Tune a little off and it spins, but the sound hardly changes, because AM only reads the length. In the " +
      "close-up, look for a tall carrier spike with matching sidebands either side. The carrier itself carries no sound.",
  },
  usb: {
    heading: "USB (upper sideband)",
    body:
      "There's no carrier, so the arrow has no resting place and the constellation is a shifting cloud. All the energy " +
      "sits above the tuned frequency. Being off by a little shifts the pitch of a voice: too high sounds like a " +
      "cartoon, too low sounds like a growl. Fine-tune until it sounds natural. Common above 10 MHz.",
  },
  lsb: {
    heading: "LSB (lower sideband)",
    body:
      "Like USB, but all the energy sits below the tuned frequency. Hams usually use it below 10 MHz. Listen as USB " +
      "and you'll only hear what happens to be on the other side, usually noise.",
  },
  cw: {
    heading: "CW (Morse)",
    body:
      "The arrow switches ON and OFF. A carrier tuned exactly would make no sound, so the app adds 700 Hz, which is why " +
      "the arrow spins 700 turns a second here while it's on. In the close-up the carrier itself sits on the tuner. " +
      "FSK data looks different: it never goes quiet, it just hops between two frequencies.",
  },
  nfm: {
    heading: "NFM (narrow FM)",
    body:
      "The sound is in the arrow's SPIN SPEED. Its length holds roughly constant, so the constellation is a ring. Each " +
      "audio sample comes from how far the arrow turned since the last pair. Weak FM turns to loud hiss, not quiet sound.",
  },
  wfm: {
    heading: "WFM (wide FM)",
    body:
      "The same idea as NFM with a far wider swing, which is why it needs 200 kHz of the band and 240,000 pairs a second. " +
      "Five pairs lead to each audio sample. The traces look solid at full speed: try 1/1000 to see the arrow move.",
  },
  audio: {
    heading: "Audio recording",
    body:
      "This file holds sound, not IQ. It was demodulated before it was saved, so the arrow and the radio signal are gone " +
      "and can't be recovered. The picture above is a spectrogram of the sound: 0 Hz at the left, not radio frequencies. " +
      "To keep the signal itself next time, set the SDR++ Recorder module to Baseband.",
  },
};

const ARTIFACTS: Note[] = [
  {
    heading: "A spike at the very center",
    body:
      "The DC spike. Some radios leak a little signal right at the frequency they're tuned to. It's the radio, not a " +
      "station. It stays at the center of the band however the file is played.",
  },
  {
    heading: "A coarse grid of dots",
    body:
      "The RTL-SDR's converter has only 8 bits, so raw pairs land on 256 levels in each of I and Q. It shows up on weak " +
      "signals. Tuned pairs look smooth, because filtering many pairs together fills in values between the steps.",
  },
  {
    heading: "Peak at or near 0 dBFS, or strange signals everywhere",
    body:
      "Overload. The radio's input ran out of range, which clips the waveform and creates false signals across the band. " +
      "It can't be fixed afterward, but less gain on the next recording will avoid it.",
  },
  {
    heading: "A steady whine",
    body:
      "Often two signals beating together. The pitch you hear is the distance between them, so a signal 3 kHz from the " +
      "carrier gives a 3 kHz whine. Look for a second spike in the close-up, then narrow the bandwidth to shut it out.",
  },
  {
    heading: "dBFS levels",
    body:
      "Decibels relative to the largest value the file can hold, so 0 is the top. They aren't calibrated to real-world " +
      "signal power, and comparing recordings made at different gains won't mean much.",
  },
];

export function notesFor(mode: string): { main: Note; others: Note[] } {
  return { main: MODE_NOTES[mode] ?? MODE_NOTES.am, others: mode === "audio" ? [] : ARTIFACTS };
}
