def read_fasta(file_contents):
    name_list = []
    sequence_list = []
    _seq = []
    _name = []

    #Handles both methods

    #List
    if type(file_contents) is list:
        for line in file_contents:
            line = str(line) #Bug with empty comments
            if line.startswith('>'):
                name_list.append(line.rstrip('\n').replace('\r',''))
                if _seq != []:
                    sequence_list.append(''.join(_seq))
                _seq = []
            else:
                _seq.append(line.rstrip('\n').upper())
        sequence_list.append(''.join(_seq).upper()) #upper case
        return name_list, sequence_list

    #String
    elif type(file_contents) is str:
        file_contents = file_contents.split('\n')
        for line in file_contents:
            if line.startswith('>'):
                name_list.append(line)
                if _seq != []:
                    sequence_list.append(''.join(_seq))
                    _seq = []
            else:
                _seq.append(line)

        sequence_list.append(''.join(_seq))        
        return name_list, sequence_list

    else:
        print('error')
        pass #Error, not list or string
