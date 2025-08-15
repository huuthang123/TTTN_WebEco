# coding: utf-8
#!/usr/bin/env python3

import argparse
import lxml.etree as ET
import os
import regex

# arguments setting 
parser = argparse.ArgumentParser()
parser.add_argument('--lcode', help='ISO 639-1 code of target language. See `lcodes.txt`.')
parser.add_argument('--max_corpus_size', type=int, default=1000000000, help='the maximum size of the corpus.')
args = parser.parse_args()

lcode = args.lcode
max_corpus_size = args.max_corpus_size
fname = "wikimania2014wiki-20250720-pages-articles-multistream.xml"

# Optional language-specific libraries
if lcode == 'ko':
    from konlpy.tag import Kkma
    kkma = Kkma()
elif lcode == 'ja':
    import MeCab
    mecab = MeCab.Tagger("-Owakati")
elif lcode == 'zh':
    import jieba
elif lcode == 'vi':
    from pyvi.pyvi import ViTokenizer
elif lcode == 'th':
    import pythai

def clean_text(text):
    if text is None:
        return ""
    
    text = regex.sub("(?s)<ref>.+?</ref>", "", text)
    text = regex.sub("(?s)<[^>]+>", "", text)
    text = regex.sub("&[a-z]+;", "", text)
    text = regex.sub("(?s){{.+?}}", "", text)
    text = regex.sub("(?s){.+?}", "", text)
    text = regex.sub("(?s)\\[\\[([^]]+\\|)", "", text)
    text = regex.sub("(?s)\\[\\[([^]]+\\:.+?]])", "", text)

    text = regex.sub("[']{5}", "", text)
    text = regex.sub("[']{3}", "", text)
    text = regex.sub("[']{2}", "", text)

    if lcode == 'ko':
        text = regex.sub(r"[^ \r\n\p{Hangul}.?!]", " ", text)
    elif lcode == 'ja':
        text = regex.sub(r"[^\r\n\p{Han}\p{Hiragana}\p{Katakana}ー。！？]", "", text)
    elif lcode == 'zh':
        text = regex.sub(r"[^\r\n\p{Han}。！？]", "", text)
    elif lcode == 'th':
        text = regex.sub(r"[^ \r\n\p{Thai}.?!]", " ", text)
    elif lcode == 'ru':
        text = regex.sub(r"[^ \r\n\p{Cyrillic}.?!\-]", " ", text)
        text = text.lower()
    elif lcode == 'hi':
        text = regex.sub(r"[^ \r\n\p{Devanagari}.।?!\-]", " ", text)
    elif lcode == 'bn':
        text = regex.sub(r"[^ \r\n\p{Bengali}.।?!\-]", " ", text)
    elif lcode == 'de':
        text = regex.sub(r"[^ \r\n\p{Latin}\-'‘’.?!]", " ", text)
    else:
        text = regex.sub(r"[^ \r\n\p{Latin}\-'‘’.?!]", " ", text)
        text = text.lower()

    text = regex.sub(r"[ ]{2,}", " ", text)
    return text.strip()

def sentence_segment(text):
    if lcode in ['ja', 'zh']:
        sents = regex.split(r"([。！？])?[\n]+|[。！？]", text)
    elif lcode == 'th':
        sents = text.split("[\n]+")
    elif lcode in ['hi', 'bn']:
        sents = regex.split(r"([.।?!])?[\n]+|[.।?!] ", text)
    elif lcode == 'de':
        sents = regex.split(r"([.?!])?[\n]+|[.?!] ", text)
        sents = [sent[0].lower() + sent[1:] for sent in sents if sent and len(sent) > 1]
    else:
        sents = regex.split(r"([.?!])?[\n]+|[.?!] ", text)
    return [s for s in sents if s and s.strip()]

def word_segment(sent):
    if lcode == 'ko':
        return [word for word, _ in kkma.pos(sent)]
    elif lcode == 'ja':
        parsed = mecab.parse(sent)
        return parsed.strip().split() if parsed else []
    elif lcode == 'th':
        return pythai.split(sent)
    elif lcode == 'vi':
        return ViTokenizer.tokenize(sent).split()
    elif lcode == 'zh':
        return list(jieba.cut(sent, cut_all=False))
    else:
        return sent.split()

def build_corpus():
    output_path = f"data/{lcode}.txt"
    input_path = f"data/{fname}"

    # 🛠 Đặt đúng namespace theo file XML bạn cung cấp
    ns = "{http://www.mediawiki.org/xml/export-0.11/}"

    with open(output_path, 'w', encoding='utf-8') as fout:
        i = 1
        for _, elem in ET.iterparse(input_path, tag=ns + "text"):
            text = elem.text
            try:
                cleaned = clean_text(text)
                sents = sentence_segment(cleaned)
                for sent in sents:
                    words = word_segment(sent)
                    if len(words) > 3:  # giảm điều kiện để ghi nhiều dòng hơn
                        fout.write(" ".join(words) + "\n")
            except Exception:
                pass
            elem.clear()
            if i % 1000 == 0:
                print(f"{i} articles parsed...", flush=True)
                if os.path.getsize(output_path) > max_corpus_size:
                    break
            i += 1

if __name__ == "__main__":
    build_corpus()
    print("✅ Corpus build complete.")
