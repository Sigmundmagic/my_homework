class rangeForChromosome(): # класс диапазон для хромосомы
    #реализация класса
    def setNumberChromosome(self,numberChromosome):#обрезаем строку, чтобы оставить только номер
        assert type(numberChromosome) == str and numberChromosome != '' and numberChromosome != None and numberChromosome.find('chr') != -1 #проверка на то что переменная не равна пустоте, не равна пустой строке и имеет 'chr'
        numberChromosome = numberChromosome[numberChromosome.find('chr') + 3:]# здесь удаляется 'chr' из строки
        self.numberChromosome = numberChromosome # присваиваем переменной класса значение
    def __init__(self,numberChromosome,beginRead,endRead): # конструктор класса принимает в качестве входных параметров numberChromosome,beginRead,endRead 
        self.setNumberChromosome(numberChromosome)
        self.beginRead = beginRead
        self.endRead = endRead
    def __init__(self,string):
        self.string = string
        self.setNumberChromosome(string.split('\t')[0].strip())
        self.beginRead = int(string.split('\t')[1].strip())
        self.endRead =int(string.split('\t')[2].strip())

    def inRange(self,value):
        assert type(value) is int
        if value >= self.beginRead and value <= self.endRead:
            return True
        else:
            return False

class strFormVCFTable():
    def __init__(self,string):
        self.string = string
        self.CHROM = string.split('\t')[0].strip()
        self.POS = int(string.split('\t')[1].strip())


def isHeader(string):
    string = string.strip()
    if len(string) < 2:
        return False
    if string[0] == '#' and string[1] != '#':
        return True
    return False

def isComment(string):
    string = string.strip()
    if len(string) < 2:
        return False
    if string[0] == '#' and string[1] == '#':
        return True
    return False

def addToResultWithCheck(line,itemFromBED):
    global listForResult
    if isHeader(line) == True or isComment(line) == True or line == '':
        return
    itemVCF = strFormVCFTable(line)
    for item in listForResult:
        if item.CHROM == itemVCF.CHROM and item.POS == itemVCF.POS:
            return
    if itemVCF.CHROM != itemFromBED.numberChromosome or itemFromBED.inRange(itemVCF.POS) == False:
        return
    listForResult.append(itemVCF)

def setHeaderString(line):
    global headerString
    if isHeader(line) == True:
        headerString = line

def findStrInVCF(itemFromBED):
    fileVCF = open('2.vcf','r')
    line = fileVCF.readline()
    setHeaderString(line)
    addToResultWithCheck(line,itemFromBED)
    while line != '':
        line = fileVCF.readline()
        setHeaderString(line)
        addToResultWithCheck(line,itemFromBED)
    fileVCF.close()



headerString = ''
listForResult = []
f = open('1.bed','r')
line = f.readline()
findStrInVCF(rangeForChromosome(line))
while line != '':
    line = f.readline()
    if line != '':
        findStrInVCF(rangeForChromosome(line))
f.close()

linesForWrite = []
linesForWrite.append(headerString)
for item in listForResult:
    print('CHROM ' + item.CHROM + '; POS ' + str(item.POS))
    linesForWrite.append(item.string)
open('result.vcf','w').writelines(linesForWrite)
