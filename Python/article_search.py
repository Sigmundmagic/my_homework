from Bio import Entrez
from Bio import Medline
Entrez.email = '' # желательно указать свой email
def getGene(author):
    try:
        print('please wait, loading data')
        with Entrez.esearch(db="pubmed", term=author+"[AUTHOR]",retmax = 5) as h1:
            with Entrez.efetch(db="pubmed", id=Entrez.read( h1 )["IdList"], retmode="text", rettype="medLine") as h2:
                listOfArticles = Medline.parse( h2 ) # коллекция статей принадлежащих автору listOfArticles - это итерируемый объект
                keywords = [] # ключевые слова из статей автора 
                for item in listOfArticles: # извлекаем из итерируемого объекта информацию о статьях
                    if 'OT' in item:  
                        for keyword in item['OT']:
                            if keyword in keywords:
                                continue
                            keywords.append(keyword) # добавляем только уникальные ключевые слова
            print('1 of 3 completed')
            listOfArticlesByID = []# id статей по ключевым словам за год 
            for keyword in keywords:
                with Entrez.esearch(db="pubmed", term = keyword, datetype = 'pdat', reldate = 365, retmax = 5) as h1:
                    tmpListId = Entrez.read(h1)['IdList']
                    for item in tmpListId:
                        if item not in listOfArticlesByID:
                            listOfArticlesByID.append(item)
            print('2 of 3 completed')
            listForGene = []
            with Entrez.elink(dbfrom="pubmed", linkname='pubmed_gene', id=listOfArticlesByID) as h1:
                tmpList = Entrez.read(h1) # получаем массив с данными по генам
                for item in tmpList:
                    if item['LinkSetDb']: 
                        for itemForArray in item['LinkSetDb'][0]['Link']:
                            if itemForArray['Id'] not in listForGene:
                                listForGene.append(itemForArray['Id'])
            return listForGene     
    except IOError :
        print('server not responding please try again later')
    except Exception as ex:
        print('Exception in getGene ', ex)
    return []
    
if __name__ == "__main__":
    print('Gene ',getGene(input('enter the first and last name of the author of the article:')))
