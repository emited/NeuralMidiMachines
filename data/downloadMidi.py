import os
import urllib2
import bs4



url = "http://classtab.org"

soup = bs4.BeautifulSoup(urllib2.urlopen(url).read(), "lxml")


links = soup.find_all("a")
links = filter(lambda x : x.has_attr("href"), links)
links = filter(lambda x : x["href"].endswith(".mid"),links)

for l in links:
	print(l)
	os.system("wget -qO- "+url+"/"+str(l["href"])+" > midi/"+str(l["href"]))
