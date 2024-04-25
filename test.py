text = ''
with open('../SB-SAT/stimuli/reading screenshot/reading-dickens-2.txt') as f:
    for line in f:
        if line=='\n':
            text += '\n'
        else:
            text += line[:-1] + ' '
text = text[:-1]
print(text)

print(text.split())