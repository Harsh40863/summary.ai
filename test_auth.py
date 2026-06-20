import requests
import json

url_login = "https://summaryai-production-bc85.up.railway.app/auth/login"
res = requests.post(url_login, json={
    "email": "testuser99@example.com",
    "password": "password123"
})
token = res.json().get("access_token")

url_me = "https://summaryai-production-bc85.up.railway.app/auth/me"
headers = {"Authorization": f"Bearer {token}"}
res_me = requests.get(url_me, headers=headers)
print("Auth Me:", res_me.status_code, res_me.text)
