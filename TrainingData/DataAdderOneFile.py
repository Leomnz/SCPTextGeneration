from bs4 import BeautifulSoup
import requests
import io
f = io.open("scp.txt", "a", encoding="utf-8")
for x in range(99):
    URL = 'http://www.scpwiki.com/scp-'+'{0:03}'.format(x)
    print("Next Url: "+URL)
    content = requests.get(URL)
    removecontent = ["Source Link:","License:","Author:","Licensing Disclosures","For more information, see Licensing Guide.","from the SCP Wiki","Cite this page as:","«","»",".jpg","about on-wiki content"]
    soup = BeautifulSoup(content.text, 'html.parser')
    k = soup.find('span', attrs={"class": "number prw54353"})
    if k is not None:
        b = k.text[0]
        k = int(k.text[1:])
        if b=="-":
            k = k*-1
        if k>50:
            for div_tag in soup.findAll('div', attrs={"id": "page-content"}):
                for p_tag in div_tag.find_all('p'):
                    h = 0
                    for x in removecontent:
                        if x in p_tag.text:
                            h = 1
                    if h == 0:
                        f.write(p_tag.text+"\n")
f.close()
