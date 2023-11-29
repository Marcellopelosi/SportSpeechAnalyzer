def text_to_sentences(text):
    text = text.replace("!", ".")
    strings = [(sentence + ".").strip() for sentence in text.split(".")]
    merged_sentences = []
    current_sentence = ''

    for string in strings:
        # Combine strings until the word count reaches 20-40
        if len(current_sentence.split()) + len(string.split()) <= 40:
            current_sentence += ' ' + string.strip()
        else:
            # If the word count exceeds 40, start a new sentence
            if len(current_sentence.split()) >= 20:
                merged_sentences.append(current_sentence.strip())
            current_sentence = string.strip()

    # Append the last sentence
    merged_sentences.append(current_sentence.strip())

    return merged_sentences
