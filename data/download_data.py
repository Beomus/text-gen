import urllib.request
import re
from bs4 import BeautifulSoup


def parse_and_fix_quotes(input):
    text = input.getText()
    text = re.sub('[\'\"`“”‘’]', '\"', text)
    text = re.sub('(?<=\w)[\"](?=\w)', '\'', text)
    text = re.sub('[\n\t]', ' ', text)
    return text



def load_from_url(num):
    fp = urllib.request.urlopen(f"http://ae-lib.org.ua/texts-c/tolkien__the_lord_of_the_rings_{num}__en.htm")
    mybytes = fp.read()
    mystr = mybytes.decode("cp1252")

    soup = BeautifulSoup(mystr, 'html.parser')
    p_tags = soup.find_all('p')

    start = [109, 28, 27][num-1]
    end = [3955, 2942, 2449][num-1]
    p_tags_processed_text = list(map(parse_and_fix_quotes, p_tags))[start:end]

    full_text = " ".join(p_tags_processed_text)
    #Fellowship Start: 109, End: 3955
    #Two towers Start: 28,  End: 2942
    #ROTK       Start: 27,  End: 2449
    return full_text


if __name__ == "__main__":
    for i in range(1, 4):
        full_text  = load_from_url(i)
        with open(f"data/book_{i}.txt", "w") as text_file:
            text_file.write(full_text)
