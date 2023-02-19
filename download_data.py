import gdown


print("Downloading file 1/2...")
url = 'https://drive.google.com/uc?id=1kiGeY8Ymv5e9YN-Ie6ZvkQWMzU3Fnq5M'
output = 'dataset.zip'
gdown.download(url, output, quiet=False)

print("Downloading file 2/2...")
url = 'https://drive.google.com/uc?id=1Ar6G7W2hUj36qrRHThk_Y3HHfQQWFFO-'
output = 'models.zip'
gdown.download(url, output, quiet=False)