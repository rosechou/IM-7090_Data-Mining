# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
#import packages
import csv
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import math
from joblib import dump, load


# %%
#data normalization
def normalize(data):
    data_ = []
    for i in range(len(data)):
        row = []
        for j in range(len(data[i])):
            if j == 2:
                row.append(int(data[i][j])/12)
            elif j == 3:
                val = (float(data[i][j]) +60)/60
                row.append(val)
            elif j == 10:
                row.append(int(round(float(data[i][j])))/200)
            else:
                row.append(float(data[i][j]))
        data_.append(row)
    return data_

# %%
#load centroids from step2 clustering results
def getCentroids():
    cs = []
    with open("centroids_3.csv", "r") as csvfile:
        rows = csv.reader(csvfile)
        for row in rows:
            drow = []
            for col in row:
                drow.append(float(col))
            cs.append(drow)
    #cs = np.array(cs)
    return cs


# %%
#每個sample使用features如下，共11個。
#['danceability','energy','key', 'loudness', 'mode', 'speechiness', 'acousticness','instrumentalness',
# 'liveness','valence','tempo']
#classify a feature vector. x must be a 1x11 vector(11 features)
def classify(centroids,x):
    minDis = 0
    belong = 0
    for i in range(len(centroids)):
        val = 0
        for j in range(len(centroids[0])):
            val += math.pow(x[j]-centroids[i][j],2)
        if i == 0:
            minDis = val
        elif val < minDis:
            minDis = val
            belong = i
    return belong


# %%
def getDataClusters(data):
    #load pca model from step2
    pca = load("pca.joblib")
    
    #get centroids from reord
    centroids = getCentroids()
    clusterAmount = len(centroids)
    #noramlize data
    data = normalize(data)
    
    #transform data with pca model
    data = np.array(data, dtype=float)
    data = pca.transform(data)
    
    #cluster stats
    nums = []
    for i in range(clusterAmount):
        nums.append(0)
    
    #classify every sample in data and output nums of each class
    clus = []
    for i in range(len(data)):
        label = classify(centroids, data[i])
        clus.append(label)
        nums[label]+=1
        
    return clus, nums


# %%
#load data
data = []
city = []
header = []
if True:
    with open("city_result_3.csv", "r") as csvfile:
        rows = csv.reader(csvfile)
        i = 0
        for row in rows:
            if i == 0:
                header = row[2:13]
            else:
                data.append(row[2:13])
                city.append(row[1])
            i += 1
test_x = data

# %%
test_y, test_y_stats = getDataClusters(test_x)

# %%
print(test_y[:10])
print(city[:10])
print(test_y_stats)


# %%
#classify every sample in data and output nums of each class
result = []
for i in range(len(test_y)):
    a = [ city[i] , test_y[i]]
    result.append(a)
print(result)


# %%
# 寫入檔案
with open('city_music_result.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(result)


# %%
# Get city - music type count
# EX ='Taipei': {0: 9, 1: 18, 2: 24, 3: 24, 4: 25}
city_music_count = {}
count = []
for r in result:
    if r[0] in city_music_count:
        if r[1] in city_music_count[r[0]]:
            city_music_count[r[0]][r[1]] += 1
    else:
        #city_music_count[r[0]] = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0, 10:0, 11:0, 12:0, 13:0, 14:0}
        city_music_count[r[0]] = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0}
        city_music_count[r[0]][r[1]] += 1
print(city_music_count)


# %%
# Get city - music type average
# ['Taipei', 0.09, 0.18, 0.24, 0.24, 0.25]
city_music_avg = []
city_music_avg.append(['City', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
#city_music_avg.append(['City', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14'])
for r in city_music_count.items():
    avg_value = []
    avg_value.append(r[0])
    count = 0
    for a in r[1].items():
        count += a[1]
    for b in range(0,10):
        avg_value.append(r[1][b]/count)
    city_music_avg.append(avg_value)

print(city_music_avg)


# %%
# 寫入檔案
# 欄位資料：['Taipei', 0.09, 0.18, 0.24, 0.24, 0.25]
with open('city_cluster_avg_result.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(city_music_avg)


# %%
import requests
import spotipy
import spotipy.oauth2 as oauth2
import csv
import time
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
# %%

root= tk.Tk()
  
canvas1 = tk.Canvas(root, width = 800, height = 300)
canvas1.pack()

label1 = tk.Label(root, text='Discover which city list matches your spotify play list !')
label1.config(font=('Arial', 20))
canvas1.create_window(400, 50, window=label1)
   
label2 = tk.Label(root, text='Input spotify ID: ')  
canvas1.create_window(400, 80, window=label2)

# input userid
entry1 = tk.Entry (root)
canvas1.create_window(400, 100, window=entry1) 
# %%

# userid = '11158258963'
allUserTracksinPlaylists = []
def getAllUserTracksinPlaylists():
    userid = entry1.get()
    url = "https://api.spotify.com/v1/users/{}/playlists".format(userid)
    headers = {
            'Accept': 'application/json',
            'Content-Type': 'application/json',
            # need to get new token to run program
            "Authorization": "Bearer BQBpRchegSLF3RVXe681qV2BJkZTIB3c5QDnnPVak_9az1k8DOwW5OHIJ_6eYlpEyYjyc2ZNk-W8_o7XBsZHWucHVB_yf84pmF8UzYZRCdaCUFWD9W_YhOtvDcnOwb99h4LHjfsmlygfbyoxCjQ133yKCNPHXAq0AuLBRwN4byKUFimg8ykM"

    }
    response = requests.get(url, headers=headers)
    playlist_item = response.json()
    print(playlist_item)
    # print(type(response))
    # print(type(playlist_item["items"]))
    playlisturl = []
    for i in playlist_item["items"]:
    #     print(i["id"], i["name"], i["tracks"])
        playlisturl.append(i["tracks"]["href"])

    trackids = []
    for listurl in playlisturl:
        url  = listurl
        headers = {
                'Accept': 'application/json',
                'Content-Type': 'application/json',
                # need to get new token to run program
                "Authorization": "Bearer BQBpRchegSLF3RVXe681qV2BJkZTIB3c5QDnnPVak_9az1k8DOwW5OHIJ_6eYlpEyYjyc2ZNk-W8_o7XBsZHWucHVB_yf84pmF8UzYZRCdaCUFWD9W_YhOtvDcnOwb99h4LHjfsmlygfbyoxCjQ133yKCNPHXAq0AuLBRwN4byKUFimg8ykM"
                }
        response = requests.get(url, headers=headers)
        tracks_item = response.json()
        for i in tracks_item["items"]:
    #         print(i['track']["id"], i['track']["name"], i['track']['href'])
            trackids.append(i['track']['id'])

    # allUserTracksinPlaylists = []
    for trackid in trackids:
        url  = 'https://api.spotify.com/v1/audio-features/{}'.format(trackid)
    #     print(url)
        headers = {
                'Accept': 'application/json',
                'Content-Type': 'application/json',
                # need to get new token to run program
                "Authorization": "Bearer BQBpRchegSLF3RVXe681qV2BJkZTIB3c5QDnnPVak_9az1k8DOwW5OHIJ_6eYlpEyYjyc2ZNk-W8_o7XBsZHWucHVB_yf84pmF8UzYZRCdaCUFWD9W_YhOtvDcnOwb99h4LHjfsmlygfbyoxCjQ133yKCNPHXAq0AuLBRwN4byKUFimg8ykM"
                }
        response = requests.get(url, headers=headers)
        tracks_features = response.json()
        # print(tracks_features)
        allUserTracksinPlaylists.append(tracks_features)
        time.sleep(0.1)

    print(allUserTracksinPlaylists[0])
    print(len(allUserTracksinPlaylists))

# getAllUserTracksinPlaylists()


# %%
allUserTracksinPlaylists


# %%
#load data
def loadData():
    playlistdata = []

    for row in allUserTracksinPlaylists:
        playlistdata.append(list(row.values())[0:11])

    print(playlistdata)
    return playlistdata


# %%
#classify every sample in data and output nums of each class
def classifyUserData():
    playlistdata = loadData()
    test_y, test_y_stats = getDataClusters(playlistdata)
    return test_y, test_y_stats


# %%
# Get avg vector
def getPlaylistAvgVec():
    test_y, test_y_stats = classifyUserData()
    playlist_avg = []
    for b in range(0,10):
        playlist_avg.append(test_y_stats[b]/len(test_y))
    print(playlist_avg)
    return playlist_avg


# %%
from scipy.spatial import distance
def compareData():
    compare = {}
    for row in city_music_avg:
        if(row[0] == 'City'):
            continue
        value = distance.euclidean(getPlaylistAvgVec(), row[1:11])
        compare[row[0]] = value
    print(compare)
    return compare


# %%
# put 3 the most similar to city and score list 
def showResult():
    compare = compareData()
    min(compare, key=compare.get)
    result_city = []
    compare = sorted(compare.items(), key=lambda x:x[1], reverse=True)
    print(compare[0:4])
    result_city = []
    result_score = []
    for i in compare[0:3]:
        result_city.append(i[0])
        result_score.append(i[1])
    result_score = [ '%.3f' % elem for elem in result_score ]

    print(result_city)
    print(result_score)
    return result_city, result_score

# %%
# draw charts
def create_charts():
    result_city, result_score = showResult()
    playlist_avg = getPlaylistAvgVec()

    global bar1
    global pie2

    figure1 = Figure(figsize=(4,3), dpi=100) 
    subplot1 = figure1.add_subplot(111) 
    xAxis = [0,1,2,3,4,5,6,7,8,9] 
    yAxis = [playlist_avg[0],playlist_avg[1],playlist_avg[2],playlist_avg[3],playlist_avg[4],playlist_avg[5],playlist_avg[6],playlist_avg[7],playlist_avg[8],playlist_avg[9]] 
    x_labels = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
    subplot1.bar(xAxis,yAxis, color = 'lightsteelblue') 
    # subplot1.title('Your play list type')
    # subplot1.xticks(x, x_labels)
    bar1 = FigureCanvasTkAgg(figure1, root) 
    bar1.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=0)
      
    figure2 = Figure(figsize=(4,3), dpi=100) 
    subplot2 = figure2.add_subplot(111) 
    labels2 = result_city
    pieSizes = result_score
    my_colors2 = ['lightblue','lightsteelblue','silver']
    explode2 = (0, 0.1, 0)  
    subplot2.pie(pieSizes, colors=my_colors2, explode=explode2, labels=labels2, autopct='%1.1f%%', shadow=True, startangle=90) 
    subplot2.axis('equal')  
    pie2 = FigureCanvasTkAgg(figure2, root)
    pie2.get_tk_widget().pack()

def clear_charts():
    bar1.get_tk_widget().pack_forget()
    pie2.get_tk_widget().pack_forget()
            
button_fetchdata = tk.Button (root, text='    Fetch data    ', command=getAllUserTracksinPlaylists, bg='lightsteelblue2', font=('Arial', 11, 'bold'))
canvas1.create_window(400, 140, window=button_fetchdata)

button1 = tk.Button (root, text=' Create Charts ',command=create_charts, bg='palegreen2', font=('Arial', 11, 'bold')) 
canvas1.create_window(400, 180, window=button1)

button2 = tk.Button (root, text='  Clear Charts  ', command=clear_charts, bg='lightskyblue2', font=('Arial', 11, 'bold'))
canvas1.create_window(400, 220, window=button2)

button3 = tk.Button (root, text='Exit Application', command=root.destroy, bg='lightsteelblue2', font=('Arial', 11, 'bold'))
canvas1.create_window(400, 260, window=button3)
 
root.mainloop()


# %%


