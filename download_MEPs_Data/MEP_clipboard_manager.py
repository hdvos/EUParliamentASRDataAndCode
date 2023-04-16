import pyperclip
import pandas
import re


def process_clipboard(content):
    content = content.splitlines()
    new_content = []
    for line in content:
        line = line.strip()
        line = line.split('–')
        name = line[0].strip()
        name = re.sub(r'\[[^\]]+\]', '', name)
        name = re.sub(r'\([^)]+\)', '', name)
        other = line[1].strip() if len(line) >= 2 else None
        other = re.sub(r'\[[0-9]+\]', '', other) if other else None
        
        name = name.split(' ')
        firstname = name[0]
        lastname = ' '.join(name[1:])

        outputline = f"{firstname}\t{lastname}\t\t\t\t\t{other if other else ''}"
        new_content.append(outputline)

    return '\n'.join(new_content)



while input("Press enter to continue. Enter X to quit: ").lower() != 'x':
    clipboard_content = pyperclip.paste()    
    new_content = process_clipboard(clipboard_content)
    print(new_content)
    pyperclip.copy(new_content)
    
