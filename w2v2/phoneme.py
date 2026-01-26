import json
from functools import reduce
from operator import add
from lib.logger import console

MAPPING = [
    ("क", "kə"), # phonemes: k and ə
    ("ख", "kʰə"), # phonemes: kʰ and ə
    ("ग", "gə"), # phonemes: g and ə
    ("घ", "gʰə"), # phonemes: gʰ and ə
    ("ङ", "ŋə"), # phonemes: ŋ and ə
    ("च", "cə"), # phonemes: c and ə
    ("छ", "cʰə"), # phonemes: cʰ and ə
    ("ज", "ɟə"), # phonemes: ɟ and ə
    ("झ", "ɟʰə"), # phonemes: ɟʰ and ə
    ("ञ", "ɲə"), # phonemes: ɲ and ə
    ("ट", "ʈə"), # phonemes: ʈ and ə
    ("ठ", "ʈʰə"), # phonemes: ʈʰ and ə
    ("ड", "ɖə"), # phonemes: ɖ and ə
    ("ढ", "ɖʰə"), # phonemes: ɖʰ and ə
    ("ण", "ɳə"), # phonemes: ɳ and ə
    ("त", "t̪ə"), # phonemes: t̪ and ə
    ("थ", "t̪ʰə"), # phonemes: t̪ʰ and ə
    ("द", "d̪ə"), # phonemes: d̪ and ə
    ("ध", "d̪ʰə"), # phonemes: d̪ʰ and ə
    ("न", "nə"), # phonemes: n and ə
    ("प", "pə"), # phonemes: p and ə
    ("फ", "pʰə"), # phonemes: pʰ and ə
    ("ब", "bə"), # phonemes: b and ə
    ("भ", "bʰə"), # phonemes: bʰ and ə
    ("म", "mə"), # phonemes: m and ə
    ("य", "jə"), # phonemes: j and ə
    ("र", "rə"), # phonemes: r and ə
    ("ल", "lə"), # phonemes: l and ə
    ("व", "və"), # phonemes: v and ə
    ("श", "ɕə"), # phonemes: ɕ and ə
    ("ष", "ʂə"), # phonemes: ʂ and ə
    ("स", "sə"), # phonemes: s and ə
    ("ह", "ɦə"), # phonemes: ɦ and ə
    ("क्ष", "kʃə"), # phonemes: k, ʃ and ə
    ("त्र", "t̪ɾə"), # phonemes: t̪, ɾ and ə
    ("ज्ञ", "gjə"), # phonemes: gj and ə
    ("श्र", "ɕcə"), # phonemes: ɕ, c and ə
    ("क़", "qə"), # phonemes: q and ə
    ("ख़", "xə"), # phonemes: x and ə
    ("ग़", "ɣə"), # phonemes: ɣ and ə
    ("ज़", "zə"), # phonemes: z and ə
    ("फ़", "fə"), # phonemes: f and ə
    ("ड़", "ɽə"), # phonemes: ɽ and ə
    ("ड़", "ɽə"), # phonemes: ɽ and ə
    ("ढ़", "ɽʱə"), # phonemes: ɽʱ and ə
    ("ढ़", "ɽʱə"), # phonemes: ɽʱ and ə
    # Inherent schwa + virama/vowel signs
    ("ə्", ""),  # virama (halant) removes inherent schwa
    ("əा", "ɑː"), # phonemes: ɑː
    ("əि", "i"), # phonemes: i
    ("əी", "iː"), # phonemes: iː
    ("əु", "u"), # phonemes: u
    ("əू", "uː"), # phonemes: uː
    ("əृ", "ɹ̩"), # phonemes: ɹ̩
    ("əे", "eː"), # phonemes: eː
    ("əै", "aːi"), # phonemes: aː and i
    ("əो", "oː"), # phonemes: oː
    ("əौ", "aːu"), # phonemes: aː and u
    # Nasalization / visarga
    ("əं", "ⁿ"), # phonemes: ⁿ
    ("əँ", "ⁿ"), # phonemes: ⁿ
    ("ं", "ⁿ"), # phonemes: ⁿ
    ("ँ", "ⁿ"), # phonemes: ⁿ
    ("əः", "h"), # phonemes: h
    # Independent vowels
    ("अ", "ə"), # phonemes: ə
    ("आ", "ɑː"), # phonemes: ɑː
    ("इ", "i"), # phonemes: i
    ("ई", "iː"), # phonemes: iː
    ("उ", "u"), # phonemes: u
    ("ऊ", "uː"), # phonemes: uː
    ("ए", "eː"), # phonemes: eː
    ("ऐ", "aːi"), # phonemes: aː and i
    ("ओ", "oː"), # phonemes: oː
    ("औ", "aːu"), # phonemes: aː and u
    ("ऋ", "ɹ̩"), # phonemes: ɹ̩
    ("ॠ", "ɹ̩ː"), # phonemes: ɹ̩ː
    ("ऌ", "l̩"), # phonemes: l̩
    ("ॡ", "l̩ː"), # phonemes: l̩ː
    ("अं", "ⁿ"), # phonemes: ⁿ
    ("अः", "h"), # phonemes: h
]

VOCABULARY = list(set(reduce(add, [item[1] for item in MAPPING])))


def devanagari_to_ipa(text: str) -> str:
    """Apply ordered, repeated replacements to convert Devanagari → IPA."""
    out = text
    for old, new in MAPPING:
        # Repeat until no more occurrences (mirrors the JS while-index loop)
        while old in out:
            out = out.replace(old, new)
    return out

def ipa_to_phoneme_list(ipa: str) -> list[str]:
    # one charcater != one phoneme.
    # input: ɑːi
    # output: [ɑː, i]
    # IMPORTANT: OUTPUT IS NOT [ɑ, ː, i]

    # Extract phonemes from MAPPING based on the comments
    # Building comprehensive phoneme list based on the comments in MAPPING
    phonemes = [
        # Multi-character phonemes first (for greedy matching)
        "ɑː", "iː", "uː", "eː", "oː", "ɹ̩ː", "l̩ː",  # Long vowels
        "aːi", "aːu",  # Diphthongs
        "kʰ", "gʰ", "cʰ", "ɟʰ", "ʈʰ", "ɖʰ", "t̪ʰ", "d̪ʰ", "pʰ", "bʰ", "ɽʱ",  # Aspirated consonants
        "t̪", "d̪",  # Dental consonants
        "kʃ", "t̪ɾ", "ɕc",  # Complex consonants
        "gj",  # Complex consonant from ज्ञ
        "ɹ̩", "l̩",  # Syllabic consonants
        # Single character phonemes
        "k", "g", "ŋ", "c", "ɟ", "ɲ", "ʈ", "ɖ", "ɳ", "n", "p", "b", "m", 
        "j", "r", "l", "v", "ɕ", "ʂ", "s", "ɦ", "q", "x", "ɣ", "z", "f", 
        "ɽ", "h", "ə", "i", "u", "ⁿ", "ɾ"
    ]
    
    # Sort by length (longest first) for greedy matching
    phonemes.sort(key=len, reverse=True)
    
    result = []
    i = 0
    
    while i < len(ipa):
        # Try to match the longest phoneme first
        matched = False
        for phoneme in phonemes:
            if ipa[i:].startswith(phoneme):
                result.append(phoneme)
                i += len(phoneme)
                matched = True
                break
        
        if not matched:
            # If no phoneme matches, add the character as-is
            # This handles any edge cases or unknown characters
            result.append(ipa[i])
            i += 1
    
    return result 

# Vocabulary should contain all possible phonemes that ipa_to_phoneme_list can output
def _get_all_phonemes():
    """Get the complete phoneme vocabulary used by ipa_to_phoneme_list"""
    return [
        # Multi-character phonemes
        "ɑː", "iː", "uː", "eː", "oː", "ɹ̩ː", "l̩ː",  # Long vowels
        "aːi", "aːu",  # Diphthongs
        "kʰ", "gʰ", "cʰ", "ɟʰ", "ʈʰ", "ɖʰ", "t̪ʰ", "d̪ʰ", "pʰ", "bʰ", "ɽʱ",  # Aspirated consonants
        "t̪", "d̪",  # Dental consonants
        "kʃ", "t̪ɾ", "ɕc",  # Complex consonants
        "gj",  # Complex consonant from ज्ञ
        "ɹ̩", "l̩",  # Syllabic consonants
        # Single character phonemes
        "k", "g", "ŋ", "c", "ɟ", "ɲ", "ʈ", "ɖ", "ɳ", "n", "p", "b", "m", 
        "j", "r", "l", "v", "ɕ", "ʂ", "s", "ɦ", "q", "x", "ɣ", "z", "f", 
        "ɽ", "h", "ə", "i", "u", "ⁿ", "ɾ"
    ]

VOCABULARY = _get_all_phonemes()

def main():
    import sys

    # Interactive mode
    console.print("Devanagari to IPA converter. Type 'exit' to quit.")
    while True:
        text = input("Devanagari: ")
        if text.lower() == "exit":
            break
        ipa = devanagari_to_ipa(text)
        phoneme_list = ipa_to_phoneme_list(ipa)
        console.print(f"IPA: {ipa}")
        console.print(f"Phonemes: {phoneme_list}")

phoneme_mapping = {}

def load_vocab_data(vocab_path: str):
    global phoneme_mapping

    vocab_dict = json.load(open(vocab_path, "r"))
    id_to_phoneme = {v: k for k, v in vocab_dict.items()}
    phoneme_mapping = vocab_dict

    return vocab_dict, id_to_phoneme

def text_to_phoneme_list(text: str) -> list[str]:

    return ipa_to_phoneme_list(devanagari_to_ipa(text))


# Reverse mapping: phoneme -> Hindi approximation
# This is a best-effort mapping since the conversion isn't always 1:1
PHONEME_TO_HINDI = {
    # Consonants (without inherent schwa for flexibility)
    "k": "क्", "kʰ": "ख्", "g": "ग्", "gʰ": "घ्", "ŋ": "ङ्",
    "c": "च्", "cʰ": "छ्", "ɟ": "ज्", "ɟʰ": "झ्", "ɲ": "ञ्",
    "ʈ": "ट्", "ʈʰ": "ठ्", "ɖ": "ड्", "ɖʰ": "ढ्", "ɳ": "ण्",
    "t̪": "त्", "t̪ʰ": "थ्", "d̪": "द्", "d̪ʰ": "ध्", "n": "न्",
    "p": "प्", "pʰ": "फ्", "b": "ब्", "bʰ": "भ्", "m": "म्",
    "j": "य्", "r": "र्", "l": "ल्", "v": "व्",
    "ɕ": "श्", "ʂ": "ष्", "s": "स्", "ɦ": "ह्", "h": "ह्",
    "q": "क़्", "x": "ख़्", "ɣ": "ग़्", "z": "ज़्", "f": "फ़्",
    "ɽ": "ड़्", "ɽʱ": "ढ़्",
    # Vowels
    "ə": "अ", "ɑː": "आ", "i": "इ", "iː": "ई", 
    "u": "उ", "uː": "ऊ", "eː": "ए", "oː": "ओ",
    "aːi": "ऐ", "aːu": "औ",
    "ɹ̩": "ऋ", "ɹ̩ː": "ॠ", "l̩": "ऌ", "l̩ː": "ॡ",
    # Nasalization
    "ⁿ": "ं",
    # Complex consonants
    "kʃ": "क्ष्", "t̪ɾ": "त्र्", "gj": "ज्ञ्", "ɕc": "श्र्",
    "ɾ": "र्",
    # Word delimiter
    "|": " ",
}


def parse_srt_file(srt_path: str) -> list[dict]:
    """Parse an SRT file and return a list of phoneme entries with timestamps."""
    entries = []
    
    with open(srt_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split by double newlines to get blocks
    blocks = content.strip().split('\n\n')
    
    for block in blocks:
        lines = block.strip().split('\n')
        if len(lines) >= 3:
            # Line 1: index
            # Line 2: timestamp
            # Line 3+: phoneme content
            try:
                index = int(lines[0])
                timestamp = lines[1]
                phoneme = '\n'.join(lines[2:]).strip()
                
                # Parse timestamp
                start_time, end_time = timestamp.split(' --> ')
                
                entries.append({
                    'index': index,
                    'start': start_time,
                    'end': end_time,
                    'phoneme': phoneme,
                })
            except (ValueError, IndexError):
                continue
    
    return entries


def srt_to_phoneme_string(srt_path: str, include_timestamps: bool = False) -> str:
    """
    Read an SRT file and return the phonemes as a space-separated string.
    Collapses consecutive identical phonemes and filters out padding.
    """
    entries = parse_srt_file(srt_path)
    
    phonemes = []
    prev_phoneme = None
    
    for entry in entries:
        phoneme = entry['phoneme']
        
        # Skip padding tokens
        if phoneme in ['<pad>', '<s>', '</s>', '<unk>']:
            continue
        
        # Collapse consecutive identical phonemes (optional - for cleaner output)
        if phoneme != prev_phoneme:
            phonemes.append(phoneme)
            prev_phoneme = phoneme
    
    return ' '.join(phonemes)


def phonemes_to_hindi(phonemes: list[str]) -> str:
    """
    Convert a list of IPA phonemes back to Hindi (Devanagari) script.
    This is a best-effort approximation since phoneme-to-text isn't always reversible.
    """
    result = []
    i = 0
    
    while i < len(phonemes):
        phoneme = phonemes[i]
        
        # Skip nasalization markers and padding
        if phoneme in ['<pad>', '<s>', '</s>', '<unk>']:
            i += 1
            continue
        
        # Handle word delimiter
        if phoneme == '|':
            result.append(' ')
            i += 1
            continue
        
        # Look up in mapping
        if phoneme in PHONEME_TO_HINDI:
            hindi = PHONEME_TO_HINDI[phoneme]
            
            # If it's a consonant (ends with virama), check if next phoneme is a vowel
            if hindi.endswith('्') and i + 1 < len(phonemes):
                next_phoneme = phonemes[i + 1]
                
                # Check if next is a vowel - replace virama with vowel matra
                vowel_matras = {
                    'ə': '',  # inherent vowel, remove virama
                    'ɑː': 'ा', 'i': 'ि', 'iː': 'ी',
                    'u': 'ु', 'uː': 'ू', 'eː': 'े', 'oː': 'ो',
                    'aːi': 'ै', 'aːu': 'ौ',
                    'ɹ̩': 'ृ', 'ɹ̩ː': 'ॄ',
                }
                
                if next_phoneme in vowel_matras:
                    # Remove virama and add vowel matra
                    result.append(hindi[:-1] + vowel_matras[next_phoneme])
                    i += 2
                    continue
            
            result.append(hindi)
        else:
            # Unknown phoneme, add as-is
            result.append(phoneme)
        
        i += 1
    
    # Clean up: remove trailing viramas where appropriate
    text = ''.join(result)
    
    # Replace consecutive viramas
    while '््' in text:
        text = text.replace('््', '्')
    
    return text


def srt_to_hindi(srt_path: str) -> tuple[str, str]:
    """
    Read an SRT file and return both the phoneme string and Hindi approximation.
    
    Returns:
        tuple: (phoneme_string, hindi_string)
    """
    entries = parse_srt_file(srt_path)
    
    phonemes = []
    prev_phoneme = None
    
    for entry in entries:
        phoneme = entry['phoneme']
        
        # Skip padding tokens
        if phoneme in ['<pad>', '<s>', '</s>', '<unk>']:
            continue
        
        # Collapse consecutive identical phonemes
        if phoneme != prev_phoneme:
            phonemes.append(phoneme)
            prev_phoneme = phoneme
    
    phoneme_string = ' '.join(phonemes)
    hindi_string = phonemes_to_hindi(phonemes)
    
    return phoneme_string, hindi_string