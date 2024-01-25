import glob
import pandas as pd
import locale
import datetime
from bs4 import BeautifulSoup
import re

locale.setlocale(locale.LC_TIME, "sv_SE")

fileNames = glob.glob('Assignment2/kungalv_slutpriser/*.html')
df = pd.DataFrame(columns = ['Date of sale', 'Address', 'Location of estate', 'Habitable area', 'Not habitable area', 'Number of rooms', 'Plot area', 'Price'])

def extractDate(result):
    return datetime.datetime.strptime(result.replace('Såld', '').strip(), '%d %B %Y')

def extractAdress(result):
    return result.strip()

def extractEstateLocation(result):
    return ' '.join(result.split())
    
def extractBoarea(result):
    return result.split()[0] if len(result.split()) != 0 else ''

def extractBiarea(result):
    return result.replace('+', '').replace('m²', '').strip()

def extractRoomNumber(result):
    result = re.findall(r'\d+\s+rum', result)
    if len(result) != 0:
        result = re.sub(r'(\d+)\s+rum', r'\1', result[0])
    else:
        result = ''
    return result

def extractPlotArea(result):
    return result.replace('m²', '').replace('tomt', '').strip()

def extractPrice(result):
    return result.replace('Slutpris', '').replace('kr', '').strip()

def toNumber(a):
    if (',' in a):
        return float(''.join(a.replace(',', '.').split()))
    elif a != '':
        return int(''.join(a.split()))
    return ''

def extractAllData(text):
    for result in text:
        date = extractDate(result.find('span', class_='hcl-label hcl-label--state hcl-label--sold-at').text)
        address = extractAdress(result.find('h2', class_='sold-property-listing__heading qa-selling-price-title hcl-card__title').text)
        estateLocation = extractEstateLocation(''.join(result.find('span', class_='property-icon property-icon--result').parent.findAll(text=True, recursive=False)))
        boarea = extractBoarea(result.find('div', class_='sold-property-listing__subheading sold-property-listing__area').text)
        biarea = extractBiarea(result.find('span', class_='listing-card__attribute--normal-weight').text if result.find('span', class_='listing-card__attribute--normal-weight') else '')
        roomNumber = extractRoomNumber(result.find('div', class_='sold-property-listing__subheading sold-property-listing__area').text)
        plotArea = extractPlotArea(result.find('div', class_='sold-property-listing__land-area').text if result.find('div', class_='sold-property-listing__land-area') else '')
        price = extractPrice(result.find('span', class_='hcl-text hcl-text--medium').text)
    
        df.loc[len(df)] = {
            'Date of sale': date,
            'Address': address,
            'Location of estate': estateLocation,
            'Habitable area': toNumber(boarea),
            'Not habitable area': toNumber(biarea),
            'Number of rooms': toNumber(roomNumber),
            'Plot area': toNumber(plotArea),
            'Price': toNumber(price) }

for fileName in fileNames:
    with open(fileName, 'r', encoding='utf-8') as file:
        html = file.read()

    soup = BeautifulSoup(html, 'html.parser')
    soup = soup.find_all('div', class_='qa-sale-card hcl-grid hcl-grid--columns-1 hcl-grid--md-columns-2')
    extractAllData(soup)

print(df.head)
df.to_csv('Assignment2/patrick/houses.csv', index=False, encoding='utf-8')
