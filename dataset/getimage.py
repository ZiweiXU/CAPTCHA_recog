import httplib

url = "mis.teach.ustc.edu.cn"

for i in range(0, 400):
    conn = httplib.HTTPConnection(url)
    conn.request("GET", "/randomImage.do?date=%271457152804424%27")
    r = conn.getresponse()
    print 'tempt {} '.format(i) + str(r.status) + ' ' + str(r.reason)
    data1 = r.read()
    fh = open('raw_image/' + str(i) + '.jpg', 'wb')
    fh.write(data1)
    fh.close()
    conn.close()
