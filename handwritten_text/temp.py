# Testando para 'ç'
letter_fs = normalize_for_filesystem("ç")
print(repr(letter_fs))
folder, folder_letter = get_folder_for_letter("ç")
print("Pasta encontrada:", folder)
