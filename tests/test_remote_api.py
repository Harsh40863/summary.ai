import requests

url_login = "https://summaryai-production-bc85.up.railway.app/auth/login"
res = requests.post(url_login, json={"email": "testuser99@example.com", "password": "password123"})
token = res.json().get("access_token")
headers = {"Authorization": f"Bearer {token}"}

# Upload a document
print("Uploading document...")
files = {'files': ('test.txt', 'This is a test document about AI and machine learning.', 'text/plain')}
res_upload = requests.post("https://summaryai-production-bc85.up.railway.app/api/upload", headers=headers, files=files)
print("Upload:", res_upload.status_code)

url_query = "https://summaryai-production-bc85.up.railway.app/api/query"

print("Testing search...")
res_search = requests.post(url_query, headers=headers, json={"query": "machine learning", "action": "search", "threshold": 0.1, "translate_to": "en"})
print("Search:", res_search.status_code, res_search.text[:200])

print("Testing explore...")
res_explore = requests.post(url_query, headers=headers, json={"query": "machine learning", "action": "explore", "threshold": 0.1, "translate_to": "en"})
print("Explore:", res_explore.status_code, res_explore.text[:200])
