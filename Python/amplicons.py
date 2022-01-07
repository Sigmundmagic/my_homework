import array
import re
import sys

class Pair:
    def __init__(self):
        self.one = ''
        self.two = ''

def complement(seq):
    basepairs = {"A": "T", "G": "C", "T": "A", "C": "G"}
    comp = ""
    for base in seq:
        comp += basepairs.get(base)
    return comp

def getDegeneratePrimer(filePathForDegeneratePrimer):
    degeneratePrimer = ''
    try:
        fileStreamForReadDegeneratePrimer = open(filePathForDegeneratePrimer,'r')
        degeneratePrimer = fileStreamForReadDegeneratePrimer.read()
        if degeneratePrimer == '':
            raise Exception('err read')
    except Exception:
        print('err read degeneratePrimer from file ' + filePathForDegeneratePrimer)
        fileStreamForReadDegeneratePrimer.close()
        sys.exit()
    fileStreamForReadDegeneratePrimer.close()
    return degeneratePrimer

# [ [A,B,C,D], [L,M,N] ] на вход в arr
def getPlotsCoincidingWithPrimer(arr,string): # -> [[начало праймера,конец праймера],...]
    pattern = ''
    for i in arr: # [ [A,B,C,D], [L,M,N] ] -> "[ABCD][LMN]" для finditer
        pattern += '['
        for item in i:
            pattern += item
        pattern += ']'
    pattern = pattern
    tmp = re.finditer(pattern,string) # возвращает итерационный объект, к которому не применимы срезы
    result = [] # массив для индексов начал и концов праймера
    for i in tmp: # в i поступает информация об одном из вариантов праймера
        result.append([i.start(),i.end()])
    return result

def getArrayByPrimer(primer):
    result = []
    table = [
        ['A','A'],
        ['G','G'],
        ['T','T'],
        ['C','C'],
        ['R','A','G'],
        ['Y','C','T'],
        ['S','G','C'],
        ['W','A','T'],
        ['K','G','T'],
        ['M','A','C'],
        ['B','C','G','T'],
        ['D','A','G','T'],
        ['H','A','C','T'],
        ['V','A','C','G'],
        ['N','A','G','T','C']]
    for item in primer: #проверка есть ли в праймере символы,которых нет в таблице: перебирает символы в праймере и сравнивает с 1 симв. в массивах внутри table 
        flag = False
        for index in range(0,len(table)):
            if table[index][0] == item:
                flag = True
        if flag == False:
            raise Exception('primer validation error in func ' + getArrayByPrimer)
    for item in primer:
        for index in range(0,len(table)):
            if table[index][0] == item:
                tmpArr = [] # массив в который добавляются все символы из подмассивов table кроме нулевого. Напр, из праймера пришёл "K" -> tmpArr получает ["G","T"]
                for i in range(1,len(table[index])):
                    tmpArr.append(table[index][i])
                result.append(tmpArr) # складываем результирующий массив
    return result

def resultStrings(firstPrimer,secondPrimer,sequenceName,sequence):
    result = [] # ["> имя посл-ти", "посл-ть", "индекс начала и посл-ть через пробел",...,"имя новой посл-ти"...]
    amplicons = [] # [[индекс начала ампликона, посл-ть ампликона],...]
    complementSequence = complement(sequence) # комплемент к полученной посл-ти
    occurrencesStringsInSequence = getPlotsCoincidingWithPrimer(getArrayByPrimer(firstPrimer),sequence) # получаем [[начало праймера, конец праймера]...]
    occurrencesStringsInComplementSequence = getPlotsCoincidingWithPrimer(getArrayByPrimer(secondPrimer),complementSequence) # тоже самое для комплемента
    i = 0
    indexOccurrenceInSequence = 0
    while indexOccurrenceInSequence < len(occurrencesStringsInSequence):
        while i < len(occurrencesStringsInComplementSequence):
            if occurrencesStringsInSequence[indexOccurrenceInSequence][0] - occurrencesStringsInComplementSequence[i][1] - 1 > 0: # ситуация при которой праймер 2 на компл. посл-ти идёт раньше, чем праймер 1 на исходной
                amplicons.append([occurrencesStringsInComplementSequence[i][1] - 1,sequence[occurrencesStringsInComplementSequence[i][1] - 1:occurrencesStringsInSequence[indexOccurrenceInSequence][0]]]) # срез ампликона из посл-ти
                i += 1
                break
            elif occurrencesStringsInComplementSequence[i][0] - occurrencesStringsInSequence[indexOccurrenceInSequence][1] - 1 > 0: # ситуация при которой праймер 1 на исх. посл-ти идёт раньше, чем праймер 2 на комплементарной
                amplicons.append([occurrencesStringsInSequence[indexOccurrenceInSequence][1] - 1,sequence[occurrencesStringsInSequence[indexOccurrenceInSequence][1] - 1:occurrencesStringsInComplementSequence[i][0]]]) # срез ампликона
                i += 1
                break
            elif occurrencesStringsInSequence[indexOccurrenceInSequence][0] - occurrencesStringsInComplementSequence[i][1] - 1 == 0 and occurrencesStringsInComplementSequence[i][0] - occurrencesStringsInSequence[indexOccurrenceInSequence][1] - 1 == 0:
                i += 1 # ампликона между праймерами не существует напр.: праймер1NNNNNNN - исходная последовательность; NNNNNNNпраймер2 - комплементарная посл-ть
                break
            i += 1
        indexOccurrenceInSequence += 1
    result.append('>' + sequenceName)
    result.append(sequence)
    for item in amplicons:
        result.append(str(item[0]) + ' ' + item[1])
    return result
            

if __name__ == '__main__':
    firstPrimer = getDegeneratePrimer(r'pr1.TXT')
    secondPrimer = getDegeneratePrimer(r'pr2.TXT')
    try:
        fo = open('result.txt','w') 
    except Exception:
        print('err create result file!')
        fo.close()
        sys.exit()

    try:
        f = open('1.fasta','r')
    except Exception:
        print('err open fasta file!')
        f.close()
        fo.close()
        sys.exit()
    try:
        line = f.readline().strip()
        if line == '':
            raise Exception('fasta file is empty')
        pair = Pair() # экземпляр класса с двумя строками (one, two)
        while True:
            if line.find('>') != -1:
                pair.one = line[line.find('>')+1:]
            else:
                pair.two = line
                tmpArray = resultStrings(firstPrimer,secondPrimer,pair.one,pair.two) # возвращает массив строк с результатом ["название","посл-ть","индексы начал ампликонов и посл-ть ампликона"]
                for line_out in tmpArray: #заполняем результирующий файл
                    fo.write(line_out + '\n')
                pair = Pair()
            line = f.readline().strip() # берём новую строку
            if len(line) == 0:
                break
        f.close()
        fo.close()
    except Exception as e:
         print(e)
         

    f.close()
    fo.close()
