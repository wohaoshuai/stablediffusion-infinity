import requests

response = requests.post("http://127.0.0.1:7860/run/outpaint", json={
	"data": [
		"hello world",
		"hello world",
		"hello world",
		1,
		7.5,
		25,
		False,
		"patchmatch",
		False,
		"disabled",
		False,
		False,
		0,
		1,
		"PLMS",
		0,
		False,
	]
}).json()
print(response)

