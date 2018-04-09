# !usr/bin/python
# coding=utf-8

import httplib, mimetypes, urlparse, json, time

import traceback

class YDMHttp(object):
	apiurl = 'http://api.yundama.com/api.php'
	username = ''
	password = ''
	appid = ''
	appkey = ''

	def __init__(self, username, password, appid, appkey):
		self.username = username
		self.password = password
		self.appid = str(appid)
		self.appkey = appkey

	def request(self, fields, files=[]):
		try:
			response = post_url(self.apiurl, fields, files)
			response = json.loads(response)
		except Exception:
			print(traceback.format_exc())
			response = None
		return response

	def balance(self):
		data = {'method': 'balance', 'username': self.username, 'password': self.password, 'appid': self.appid,
		        'appkey': self.appkey}
		response = self.request(data)
		if (response):
			if (response['ret'] and response['ret'] < 0):
				return response['ret']
			else:
				return response['balance']
		else:
			print('-9001')
			return -9001

	def login(self):
		data = {'method': 'login', 'username': self.username, 'password': self.password, 'appid': self.appid, 'appkey': self.appkey}
		response = self.request(data)
		if (response):
			if (response['ret'] and response['ret'] < 0):
				return response['ret']
			else:
				return response['uid']
		else:
			print('-9001')
			return -9001

	def upload(self, filename, codetype, timeout):
		data = {'method': 'upload', 'username': self.username, 'password': self.password, 'appid': self.appid,
		        'appkey': self.appkey, 'codetype': str(codetype), 'timeout': str(timeout)}
		file = {'file': filename}
		response = self.request(data, file)
		if (response):
			if (response['ret'] and response['ret'] < 0):
				return response['ret']
			else:
				return response['cid']
		else:
			print('-9001')
			return -9001

	def result(self, cid):
		data = {'method': 'result', 'username': self.username, 'password': self.password, 'appid': self.appid,
		        'appkey': self.appkey, 'cid': str(cid)}
		response = self.request(data)
		return response and response['text'] or ''

	def decode(self, filename, codetype, timeout):
		cid = self.upload(filename, codetype, timeout)
		if (cid > 0):
			for i in range(0, timeout):
				result = self.result(cid)
				if (result != ''):
					return cid, result
				else:
					time.sleep(1)
			print('-3003')
			return -3003, ''
		else:
			return cid, ''

######################################################################

def post_url(url, fields, files=[]):
	urlparts = urlparse.urlsplit(url)
	return post_multipart(urlparts[1], urlparts[2], fields, files)

def post_multipart(host, selector, fields, files):
	content_type, body = encode_multipart_formdata(fields, files)
	h = httplib.HTTP(host)
	h.putrequest('POST', selector)
	h.putheader('Host', host)
	h.putheader('Content-Type', content_type)
	h.putheader('Content-Length', str(len(body)))
	h.endheaders()
	h.send(body)
	errcode, errmsg, headers = h.getreply()
	return h.file.read()

def encode_multipart_formdata(fields, files=[]):
	BOUNDARY = 'WebKitFormBoundaryJKrptX8yPbuAJLBQ'
	CRLF = str('\r\n')
	L = []
	for field in fields:
		key = field
		value = fields[key]
		L.append('--' + BOUNDARY)
		L.append('Content-Disposition: form-data; name="%s"' % key)
		L.append('')
		L.append(value)
	for field in files:
		key = field
		filepath = files[key]
		L.append('--' + BOUNDARY)
		L.append('Content-Disposition: form-data; name="%s"; filename="%s"' % (key, filepath))
		L.append('Content-Type: %s' % get_content_type(filepath))
		L.append('')
		L.append(open(filepath, 'rb').read())
	L.append('--' + BOUNDARY + '--')
	L.append('')
	L = [str(i) for i in L]
	body = CRLF.join(L)
	content_type = 'multipart/form-data; boundary=%s' % BOUNDARY
	return content_type, body

def get_content_type(filename):
	return mimetypes.guess_type(filename)[0] or 'application/octet-stream'

def getIdentifyingCode(filename, codetype = 1004):
	username = ''
	password = ''
	appid = 1
	appkey = '22cc5376925e9387a23cf797cb9ba745'
	# codetype = 1005 #5
	# codetype = 1004 #4
	timeout = 60
	yundama = YDMHttp(username, password, appid, appkey)
	uid = yundama.login()
	balance = yundama.balance()
	cid, result = yundama.decode(filename, codetype, timeout)
	return result.lower().strip()
