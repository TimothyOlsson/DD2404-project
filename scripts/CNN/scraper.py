import os
import re
import multiprocessing
import requests
from bs4 import BeautifulSoup
import webbrowser
import time


def scraper_worker(arg_list):
    worker_number = arg_list[0]
    url = arg_list[1]

    user_agent = {'Referer':url,
                  'User-Agent': "Mozilla/5.0 (X11; Linux i686) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/28.0.1500.52 Safari/537.36"}
    r = requests.get(url,
                     headers = user_agent)
    try:
        r.raise_for_status()
    except Exception as e:
        print('ERROR LOADING REQUEST: ' + str(e))
        return

    # Seq
    seq_scraped = re.search(r'''Sequence</td><td class="highlight"(.*?)</tr>''', str(r.text))  # Extract seq part
    if seq_scraped is None:
        print(f'ERROR FOR WORKER {worker_number}, sequence not found')
        return
    seq_shreaded = re.findall(r'''<font color=.*?>(.*?)</font>''', seq_scraped.group(1))
    seq = ''
    for i in seq_shreaded:
        if '<br>' in i:
            i = i.replace('<br>', '')
        seq += str(i)
    print(seq)

    # Name
    name_scraped = re.search(r'''<td class="highlight">(.*?\d)&nbsp;''', r.text)  # ID always ends with digit
    if name_scraped is None:
        print(f'ERROR FOR WORKER {worker_number}, name not found')
        return
    name = name_scraped.group(1)
    print(name)

    with open(f'scraped/scrape_worker{worker_number}.fasta', 'a', encoding="utf-8") as file: # Encoding to prevent name errors
        file.write(f'>{name}\n')
        file.write(f'{seq}\n')

def main_scraper(url):
    pages = 0
    while True:
        # Note: http://www.signalpeptide.de/index.php?m=listspdb_mammalia, remove /index.php...
        url_first_part =  url.rsplit('/',1)[0]

        # Used to trick website it's a real browser
        user_agent = {'Referer':url,
                      'User-Agent': "Mozilla/5.0 (X11; Linux i686) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/28.0.1500.52 Safari/537.36"}

        # Get page
        print(f'Getting page {pages}...')
        r = requests.get(url,
                         headers = user_agent)

        # Check for errors for request
        try:
            r.raise_for_status()
        except Exception as e:
            print('ERROR LOADING REQUEST: ' + str(e))
            quit()

        soup = BeautifulSoup(r.text, 'html.parser')  # Get the html
        all_links = soup.find_all('tr', attrs={'class': 'ltabrow'})  # All links are stored in the class ltabrow
        fixed_links = []
        for worker_number, links in enumerate(all_links):
            link = re.search(r'''.href='(.*?)><a class="black"''', str(links))  # Extract one link (there are about 5 links each)
            link = link.group(1) # Extract only link
            link = link.replace('amp;', '')  # Artifacts from website
            link = url_first_part + '/' + str(link)  # Fix link for future use
            fixed_links.append([worker_number, link])

        # Pool multiprocess
        number_of_workers = len(fixed_links)  # About 50 per page --> 50 workers
        print(f'{number_of_workers} workers scraping')
        pool = multiprocessing.Pool(number_of_workers)
        pool.map(scraper_worker, fixed_links)
        pool.close()  # Close when workers are done
        pool.join()  # Wait for close

        print('Getting next page...')
        next_page = re.search(r'''align="center".*?<a class="bblack" href="(.*?)">></a>''', r.text)
        next_page = next_page.group(1)
        url = url_first_part + next_page
        pages += 1  # Count pages
        time.sleep(1.)  # Wait a bit to not overload server



if __name__ == "__main__":  # Needed for multiprocessing
    if not os.path.exists('scraped'):
        print('Creating scrape folder')
        os.makedirs('scraped')
    main_scraper('http://www.signalpeptide.de/index.php?m=listspdb_mammalia')
    print('Done')