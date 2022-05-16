from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.proxy import *
import csv
import re

class Scrapper(object):
    def __init__(self):
        PATH = "D:\chromedriver.exe"
        options = Options()
        options.page_load_strategy = 'eager'
        options.add_argument('headless')
        options.add_experimental_option('excludeSwitches', ['enable-logging'])
        user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        options.add_argument('user-agent={0}'.format(user_agent))
        #options.add_argument('proxy-server=%s' % myProxy)
        self.driver = webdriver.Chrome(PATH,options=options)
        self.action = ActionChains(self.driver)


    def load(self, link):
        #LINK WEBSITE TARUH SINI AJA VVVV
        self.driver.get(link)
        return self.driver
    
    def get_data(self):
        html = self.driver.page_source
        page_soup = BeautifulSoup(html, 'html.parser') #Mengambil element pada page pertama (Page Search)
        return page_soup

    def input(self, type, elem,x):
        xpath = f'//{type}[@{x}="{elem}"]'
        button = self.driver.find_element_by_xpath(xpath)
        try:
            button.click()
        except:
            pass
        time.sleep(5)

    def parsing_data(self, type, elem, html):
        data = html.findAll(type,elem)
        return data

    def savetocsv(self,data):
        with open('data.csv', 'w', newline='') as csvfile:
            fieldnames = ['food_name', 'food_info']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for index,_ in enumerate(data['food_name']):
                writer.writerow({'food_name': data['food_name'][index], 'food_info': data['food_info'][index]})

if __name__ == "__main__":
    data = {'food_name':[],'food_info':[]}

    scrapper = Scrapper()

    scrapper.load("https://nilaigizi.com/pencarian/advance#ctn_adv")
    scrapper.get_data()

    scrapper.input("div","col-md-3 offset-md-3 pt-3",'class')
    search_page = scrapper.get_data()

    food_name = scrapper.parsing_data('div','row text-success',search_page)
    food_info = scrapper.parsing_data('div','row text-body',search_page)

    for index,i in enumerate(food_name):
        data['food_name'].append(''.join(re.findall("\w.*\w",food_name[index].text.replace('\n',''))))
        data['food_info'].append(''.join(re.findall("\w.*\w",food_info[index].text.replace('\n',''))))

    page = re.findall('\d{1,3}',search_page.findAll('span','text-muted')[-1:][0].text)
    for i in range(int(page[1])-1):
        scrapper.input("a","Next",'aria-label')
        search_page = scrapper.get_data()
        new_page = re.findall('\d{1,3}',search_page.findAll('span','text-muted')[-1:][0].text)
        print(f'{new_page[0],new_page[1]}')
        food_name = scrapper.parsing_data('div','row text-success',search_page)
        food_info = scrapper.parsing_data('div','row text-body',search_page)

        for index,i in enumerate(food_name):
            data['food_name'].append(''.join(re.findall("\w.*\w",food_name[index].text.replace('\n',''))))
            data['food_info'].append(''.join(re.findall("\w.*\w",food_info[index].text.replace('\n',''))))

    scrapper.driver.close()

    scrapper.savetocsv(data)
    