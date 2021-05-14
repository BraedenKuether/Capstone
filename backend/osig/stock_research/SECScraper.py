import urllib.request
import urllib
from urllib.error import HTTPError
from pathlib import Path
import zipfile
import shutil
import os

#def zipdir(path, ziph):
#    for root, dirs, files in os.walk(path):
#        for file in files:
#            ziph.write(os.path.join(root,file),
#                       os.path.relpath(os.path.join(root, file),
#                                       os.path.join(path, "..")))

def createZip(cik, filename, competitors):

    #Select the ticker and destination file
    #ticker = ""
    dest = os.path.join(Path.cwd(), "downloads")
    dirct = os.path.join(dest, filename)
    os.mkdir(dirct)

    #CIK from JSON response
    #cik = "858877"

    #Add zeros to build full CIK number
    fullCik = cik
    for i in range(10 - len(cik)):
        fullCik = "0" + fullCik

    #Build search urls for 10-K & 10-Q filtered SEC page
    secSearchTenK = "https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=" + fullCik + "&type=10-K&dateb=&owner=exclude&count=10&search_text="
    secSearchTenQ = "https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=" + fullCik + "&type=10-Q&dateb=&owner=exclude&count=10&search_text="

    print(secSearchTenK)
    #Get 10-K html as a string
    opener = urllib.request.URLopener()
    opener.addheader('User-Agent', 'whatever')
    fp = urllib.request.Request(secSearchTenK, headers={'User-Agent': 'Mozilla/5.0'})
    myBytes = urllib.request.urlopen(fp).read()
    secHtmlTenK = myBytes.decode("utf8")

    #Get 10-Q html as a string
    fp = urllib.request.Request(secSearchTenQ, headers={'User-Agent': 'Mozilla/5.0'})
    myBytes = urllib.request.urlopen(fp).read()
    secHtmlTenQ = myBytes.decode("utf8")

    #Parts to later build excel download urls
    urlBeg = "https://www.sec.gov/Archives/edgar/data/" + cik + "/"
    urlEnd = "/Financial_Report.xlsx"

    #Locate the unique file number to build full excel download url
    position = 0
    strLen = 150
    length = len(secHtmlTenK)
    years = 10
    files = []

    def findSlash(tempstr):
        return tempstr.find('/')

    for i in range(years):
        tempstr = secHtmlTenK[position:length]
        tenK = tempstr.find('<td nowrap="nowrap">10-K')
        if tenK != -1:
            tenK += position
            position = (tenK + strLen)
            url = secHtmlTenK[tenK:position]

            for j in range(6):
                nextSlash = findSlash(url) + 1
                url = url[nextSlash:strLen]

            nextSlash = findSlash(url)
            url = url[0:nextSlash]
            url = urlBeg + url + urlEnd
            files.append(url)

    #print(files)

    #Get up to 10 years of 10-K data - break upon 404 error
    k = 0
    for f in files:
        try:
            temp = os.path.join(dirct, "10-K" + str(k+1) + ".xlsx")
            opener.retrieve(files[k], temp)
        except HTTPError as err:
            if err.code == 404:
                break
            else:
                raise
        k += 1

    #Locate the unique file number to build full excel download url
    position = 0
    strLen = 150
    length = len(secHtmlTenQ)
    years = 4
    files = []

    for i in range(years):
        tempstr = secHtmlTenQ[position:length]
        tenQ = tempstr.find('<td nowrap="nowrap">10-Q')
        if tenQ != -1:
            tenQ += position
            position = (tenQ + strLen)
            url = secHtmlTenQ[tenQ:position]

            for j in range(6):
                nextSlash = findSlash(url) + 1
                url = url[nextSlash:strLen]

            nextSlash = findSlash(url)
            url = url[0:nextSlash]
            url = urlBeg + url + urlEnd
            files.append(url)

    #Get up to 10 years of 10-K data - break upon 404 error
    k = 0
    for f in files:
        try:
            temp = os.path.join(dirct, "10-Q" + str(k+1) + ".xlsx")
            opener.retrieve(files[k], temp)
        except HTTPError as err:
            if err.code == 404:
                break
            else:
                raise
        k += 1

    #zip folder'
    #zipname = filename + ".zip"

    #zipf = zipfile.ZipFile(zipname, 'w', zipfile.ZIP_DEFLATED)
    #zipdir(dest, zipf)
    #zipf.close()

    shutil.make_archive(dirct, 'zip', dest, filename)

    #remove temp directory
    os.remove(dirct)

    filename = filename + ".zip"
    dirct = os.path.join(dest, filename)
    return dirct
