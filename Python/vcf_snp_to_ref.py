import array
def isHeader(string):
    string = string.strip()
    if len(string) < 2:
        return False
    if string[0] == '#' and string[1] != '#':
        return True
    return False

def replaceChar(string,index,old,new):
    if string[index] != old:
        return string
    arr = array.array('u',string)
    arr.insert(index,new)
    string = arr.tounicode()
    return string

def isComment(string):
    string = string.strip()
    if len(string) < 2:
        return False
    if string[0] == '#' and string[1] == '#':
        return True
    return False

def raplaceInlistForFasta(name,pos,REF,ALT):
    global listForFasta
    for item in listForFasta:
        if item.one == name and len(item.two) >= pos and item.two[pos] == REF:
            item.two = replaceChar(item.two,pos,REF,ALT)#item.two[pos] = ALT

class strFormVCFTable():
    def __init__(self,string):
        self.CHROM = string.split('\t')[0].strip()
        self.POS = int(string.split('\t')[1].strip())
        self.REF = string.split('\t')[3].strip()
        self.ALT = string.split('\t')[4].strip()

def updateListVCF():
    global fileVCF
    line = fileVCF.readline()
    while True:
        if isHeader(line) == True or isComment(line) == True:
            line = fileVCF.readline()
            if len(line) == 0:
                break
            continue
        itemFromVCF = strFormVCFTable(line)
        raplaceInlistForFasta(itemFromVCF.CHROM,itemFromVCF.POS,itemFromVCF.REF,itemFromVCF.ALT)
        line = fileVCF.readline()
        if len(line) == 0:
            break

def saveResultFile():
    global listForFasta
    fileResult = open('result.fasta','w')
    listForFile = []
    for item in listForFasta:
        listForFile.append('>' + item.one + '\n')
        listForFile.append(item.two + '\n')
    fileResult.writelines(listForFile)
    fileResult.close()

class Pair:
    def __init__(self):
        self.one = ''
        self.two = ''
    
listForFasta = []
f = open('1.fasta','r')
line = f.readline().strip()
pair = Pair()
while True:
    if line.find('>') != -1:
        pair.one = line[line.find('>')+1:]
    else:
        pair.two = line
        listForFasta.append(pair)
        pair = Pair()
    line = f.readline().strip()
    if len(line) == 0:
        break
f.close()

fileVCF = open('2.vcf')
updateListVCF()
fileVCF.close()
saveResultFile()
