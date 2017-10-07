import math
import collections
import time
import csv
from multiprocessing import Pool


# the method to calculate average value.
# input is a list and the output is average value
def average(list):
    if len(list) == 0:
        return 0
    else:
        return sum(list) / len(list)


# the method to get median value
# input is a list and output is median value
def median(list):
    list = sorted(list)
    if len(list) % 2 == 1:
        return list[int(len(list) / 2)]
    else:
        return (list[int(len(list) / 2) - 1] + list[int(len(list) / 2)]) / 2


# the method to calculate the standard deviation
# input is a list and output is std value
def std(list):
    avg = average(list)
    sum = 0
    for i in list:
        sum = sum + (i - avg) ** 2
    return math.sqrt(sum / len(list))


# the method calculate prediction of mean rating approach for one L1O
# the input is target user_id and item_id. and output is the prediction
def mean_item_rating(user_id, item_id):
    if item_id in user_r[user_id].keys():
        if len(item_rating[item_id]) > 1:
            l = item_rating[item_id]
            sum_mean = sum(l) - user_r[user_id][item_id]
            return sum_mean / (len(l) - 1)
        else:
            return "cant generate!"
    else:
        return "cant generate!"


# the calculation of similarity used in distance based approach
# the input is two user_id and the target user_id
def user_similarity(user1, user2, item):
    sum_sim = 0
    num = 0
    for key in (set(user_r[user1].keys()) & set(user_r[user2].keys())):
        if key == item:
            continue
        sum_sim = sum_sim + (user_r[user1][key] - user_r[user2][key]) * (user_r[user1][key] - user_r[user2][key])
        num = num + 1
    return sum_sim / num


# the calculation of distance based prediction for a target user L1O
# the input is target user_id, item_id, neighbouthood size and minimum overlap
# the output is the prediction rating value
def distance_prediction(user1, item, nsize, overlap):
    maxdiff = 16
    up = 0
    bottom = 0
    if nsize != 943:
        sim = {}
        for user in item_r[item].keys():
            if user == user1:
                continue
            if (set(user_r[user1].keys()) & set(user_r[user].keys())) == {item}:
                continue
            sim[user] = user_similarity(user1, user, item)
        sim_sorted = sorted(sim.items(), key=lambda d: d[1], reverse=False)
        for info in sim_sorted[:5]:
            up = up + user_r[info[0]][item] * (1 - sim[info[0]] / maxdiff)
            bottom = bottom + (1 - sim[info[0]] / maxdiff)
        del sim_sorted[:]
    elif overlap != 1:
        for user in item_r[item].keys():
            if user == user1:
                continue
            if (len(set(user_r[user1].keys()) & set(user_r[user].keys()))) < overlap:
                continue
            sim1 = user_similarity(user1, user, item)
            weight = 1 - (sim1 / maxdiff)
            up = up + user_r[user][item] * weight
            bottom = bottom + weight
    if bottom == 0:
        bottom = 1
    return round(up / bottom, 13)


# the method to calculate the person similarity of two user
# input is two diffreent user_id and target item_id
# the output is the value of similarity
def pearson_similarity(user1, user2, item):
    up = 0
    bottom1 = 0
    bottom2 = 0
    avg1 = (user_s[user1][0] - user_r[user1][item]) / (user_s[user1][1] - 1)
    avg2 = user_s[user2][0] / user_s[user2][1]
    for key in (set(user_r[user1].keys()) & set(user_r[user2].keys())):
        if key == item:
            continue
        up = up + (user_r[user1][key] - avg1) * (user_r[user2][key] - avg2)
        bottom1 = bottom1 + (user_r[user1][key] - avg1) * (user_r[user1][key] - avg1)
        bottom2 = bottom2 + (user_r[user2][key] - avg2) * (user_r[user2][key] - avg2)
    if bottom2 and bottom1 != 0:
        return up / (math.sqrt(bottom1) * math.sqrt(bottom2))
    else:
        return 0


# the method to calculate the prediction of Resnick approach
# input is tagert user_id, item_is, neighbourhood size and minimum overlap
# output is the prediction rating
def pearson_prediction(user1, item, nsize, overlap):
    avg1 = (user_s[user1][0] - user_r[user1][item]) / (user_s[user1][1] - 1)
    up = 0
    bottom = 0
    if nsize != 943:
        sim = {}
        for user in item_r[item].keys():
            if user == user1:
                continue
            if (set(user_r[user1].keys()) & set(user_r[user].keys())) == {item}:
                continue
            sim[user] = pearson_similarity(user1, user, item)
        sim_sorted = sorted(sim.items(), key=lambda d: d[1], reverse=True)
        for info in sim_sorted[:nsize]:
            avg2 = user_s[info[0]][0] / user_s[info[0]][1]
            up = up + (user_r[info[0]][item] - avg2) * sim[info[0]]
            bottom = bottom + abs(sim[info[0]])
        del sim_sorted[:]
    elif overlap != 1:
        for user in item_r[item].keys():
            if user == user1:
                continue
            if (len(set(user_r[user1].keys()) & set(user_r[user].keys()))) < overlap:
                continue
            sim1 = pearson_similarity(user1, user, item)
            avg2 = user_s[user][0] / user_s[user][1]
            up = up + (user_r[user][item] - avg2) * sim1
            bottom = bottom + abs(sim1)
    if bottom == 0:
        return 0
    else:
        return avg1 + up / bottom


# the method is the preparation for Resnick approach prediction using multiprocessing
# input is the start user_id, end user_id, neighbourhood size and minimum overlap
# output is the target user_id, item_id, actural rating, predict reating and its rmse value
def multi(start, end, nsize, overlap):
    rmse_user = []
    rmse_item = []
    actural_rating = []
    predict_rating = []
    rmse = []
    for user_id in range(start, end):
        for item_id in user_r[user_id].keys():
            if len(item_rating[item_id]) > 1:
                prediction = pearson_prediction(user_id, item_id, nsize, overlap)
                if prediction == 0:
                    continue
                rmse.append(abs(user_r[user_id][item_id] - prediction))
                rmse_user.append(user_id)
                rmse_item.append(item_id)
                actural_rating.append((user_r[user_id][item_id]))
                predict_rating.append(prediction)
        print("finish" + str(user_id))
    return rmse_user, rmse_item, actural_rating, predict_rating, rmse


# the method is the preparation for distance based approach prediction using multiprocessing
# input is the start user_id, end user_id, neighbourhood size and minimum overlap
# output is the target user_id, item_id, actural rating, predict reating and its rmse value
def multi_dis(start, end, nsize, overlap):
    rmse_user = []
    rmse_item = []
    actural_rating = []
    predict_rating = []
    rmse = []
    for user_id in range(start, end):
        for item_id in user_r[user_id].keys():
            if len(item_rating[item_id]) > 1:
                prediction = distance_prediction(user_id, item_id, nsize, overlap)
                if prediction == 0:
                    continue
                rmse.append(abs(user_r[user_id][item_id] - prediction))
                rmse_user.append(user_id)
                rmse_item.append(item_id)
                actural_rating.append((user_r[user_id][item_id]))
                predict_rating.append(prediction)
        print("finish" + str(user_id))
    return rmse_user, rmse_item, actural_rating, predict_rating, rmse


# start timestamp for record efficiency
start = time.clock()

# open file and initialize dictionary to save data
f = open("100k.csv")
data = f.readlines()
user_rating = collections.defaultdict(list)
item_rating = collections.defaultdict(list)
user_r = collections.defaultdict(dict)
item_r = collections.defaultdict(dict)
user_s = collections.defaultdict(dict)
rating = []

# read data from csv file
for row in data:
    row.strip()
    i = row.split(",")
    user_rating[int(i[0])].append(int(i[2]))
    item_rating[int(i[1])].append(int(i[2]))
    user_r[int(i[0])][int(i[1])] = int(i[2])
    item_r[int(i[1])][int(i[0])] = int(i[2])
    rating.append(i[2])

# get the user number, item number,rating number and rating density for task 1
user_num = len(user_rating)
item_num = len(item_rating)
rating_num = len(rating)
rating_density = rating_num / user_num / item_num

# save details about user in user_detail.csv
with open('user_detail1.csv', 'w') as f:
    f_csv = csv.writer(f, lineterminator='\n')
    for i in user_rating.keys():
        average_user = average(user_rating[i])
        median_user = median(user_rating[i])
        std_user = std(user_rating[i])
        max_user = max(user_rating[i])
        min_user = min(user_rating[i])
        row = [i, average_user, median_user, std_user, max_user, min_user]
        f_csv.writerow(row)
f.close()

# save details about item in user_detail.csv
with open('item_detail1.csv', 'w') as f:
    f_csv = csv.writer(f, lineterminator='\n')
    for i in item_rating.keys():
        average_item = average(item_rating[i])
        median_item = median(item_rating[i])
        std_item = std(item_rating[i])
        max_item = max(item_rating[i])
        min_item = min(item_rating[i])
        row = [i, average_item, median_item, std_item, max_item, min_item]
        f_csv.writerow(row)
f.close()

# the whole procedure for getting all possible rating for mean rating approach and save data in mean_rating.csv
sum_error = 0
rmse_user = []
rmse_item = []
actural_rating = []
predict_rating = []
rmse = []
acc=[]
for user_id in user_rating.keys():
    for item_id in item_rating.keys():
        if mean_item_rating(user_id, item_id) != "cant generate!":
            rmse.append(math.sqrt((user_r[user_id][item_id] - mean_item_rating(user_id, item_id)) ** 2))
            rmse_user.append(user_id)
            rmse_item.append(item_id)
            actural_rating.append((user_r[user_id][item_id]))
            predict_rating.append(mean_item_rating(user_id, item_id))

lo = 0

with open('mean_rating.csv', 'w') as f:
    f_csv = csv.writer(f, lineterminator='\n')
    while lo < len(rmse):
        row = [rmse_user[lo], rmse_item[lo], actural_rating[lo], predict_rating[lo], rmse[lo]]
        f_csv.writerow(row)
        acc.append(rmse[lo]**2)
        lo = lo + 1
    end = time.clock()
    time111 = "time: " + str(end - start) + "s"
    coverage = "coverage:" + str(len(rmse_item) / 100000)
    rrr = float(average(acc))
    rmse_re = "RMSE:" + str(math.sqrt(rrr))
    abccc = [time111, coverage,rmse_re]
    f_csv.writerow(abccc)
f.close()

# the whole procedure for getting all possible rating for distance based approach and save data in rmse_distance_multi.csv
# this method use multiprocessing by dividing 943 users into 6 group, each group is handled by a processor
rmse_user_r = []
rmse_item_r = []
actural_rating_r = []
predict_rating_r = []
rmse_r = []
if __name__ == '__main__':
    p = Pool()
    results = []
    # the neighbourhood policy can be changed here
    # nsize is for neighbourhood size, default number is 943. range is 1-943
    # overlap is for minimum overlap, default number is 1, range is 1-1682
    # this two value can only change one each time. when we change nsize, overlap need to be 1. when we change overlap, nsize need to be 943
    nsize = 943
    overlap = 200
    result = p.apply_async(multi_dis, args=(1, 160, nsize, overlap))
    results.append(result)
    result = p.apply_async(multi_dis, args=(160, 320, nsize, overlap))
    results.append(result)
    result = p.apply_async(multi_dis, args=(320, 480, nsize, overlap))
    results.append(result)
    result = p.apply_async(multi_dis, args=(480, 640, nsize, overlap))
    results.append(result)
    result = p.apply_async(multi_dis, args=(640, 800, nsize, overlap))
    results.append(result)
    result = p.apply_async(multi_dis, args=(800, 944, nsize, overlap))
    results.append(result)
    p.close()
    p.join()
    for result in results:
        rmse_user_r = rmse_user_r + result.get()[0]
        rmse_item_r = rmse_item_r + result.get()[1]
        actural_rating_r = actural_rating_r + result.get()[2]
        predict_rating_r = predict_rating_r + result.get()[3]
        rmse_r = rmse_r + result.get()[4]

lo = 0
accuracy = []
with open('rmse_distance_multi_aaa.csv', 'w') as f:
    f_csv = csv.writer(f, lineterminator='\n')
    while lo < len(rmse_r):
        row = [rmse_user_r[lo], rmse_item_r[lo], actural_rating_r[lo], predict_rating_r[lo], rmse_r[lo]]
        f_csv.writerow(row)
        accuracy.append(rmse_r[lo] * rmse_r[lo])
        lo = lo + 1
    end = time.clock()
    time111 = "time: " + str(end - start) + "s"
    coverage = "coverage:" + str(len(rmse_r) / 100000)
    rrr = float(average(accuracy))
    rmse_re = "RMSE:" + str(math.sqrt(rrr))
    print(time111, coverage, rmse_re)
    abccc = [time111, coverage, rmse_re]
    f_csv.writerow(abccc)

# saving all sum and length of users' rating for faster L1O average user rating calculation
for us in user_rating.keys():
    user_s[us] = (sum(user_rating[us]), len(user_rating[us]))

# the whole procedure for getting all possible rating for distance based approach and save data in rmse_person_multi.csv
# this method use multiprocessing by dividing 943 users into 6 group, each group is handled by a processor
rmse_user_r = []
rmse_item_r = []
actural_rating_r = []
predict_rating_r = []
rmse_r = []
if __name__ == '__main__':
    p = Pool()
    results = []
    # the neighbourhood policy can be changed here
    # nsize is for neighbourhood size, default number is 943. range is 1-943
    # overlap is for minimum overlap, default number is 1, range is 1-1682
    # this two value can only change one each time. when we change nsize, overlap need to be 1. when we change overlap, nsize need to be 943
    nsize = 943
    overlap = 200
    result = p.apply_async(multi, args=(1, 160, nsize, overlap))
    results.append(result)
    result = p.apply_async(multi, args=(160, 320, nsize, overlap))
    results.append(result)
    result = p.apply_async(multi, args=(320, 480, nsize, overlap))
    results.append(result)
    result = p.apply_async(multi, args=(480, 640, nsize, overlap))
    results.append(result)
    result = p.apply_async(multi, args=(640, 800, nsize, overlap))
    results.append(result)
    result = p.apply_async(multi, args=(800, 944, nsize, overlap))
    results.append(result)
    p.close()
    p.join()
    for result in results:
        rmse_user_r = rmse_user_r + result.get()[0]
        rmse_item_r = rmse_item_r + result.get()[1]
        actural_rating_r = actural_rating_r + result.get()[2]
        predict_rating_r = predict_rating_r + result.get()[3]
        rmse_r = rmse_r + result.get()[4]
lo = 0
accuracy = []
with open('rmse_pearson_multi_bbb.csv', 'w') as f:
    f_csv = csv.writer(f, lineterminator='\n')
    while lo < len(rmse_r):
        row = [rmse_user_r[lo], rmse_item_r[lo], actural_rating_r[lo], predict_rating_r[lo], rmse_r[lo]]
        f_csv.writerow(row)
        accuracy.append(rmse_r[lo] * rmse_r[lo])
        lo = lo + 1
    end = time.clock()
    time111 = "time: " + str(end - start) + "s"
    coverage = "coverage:" + str(len(rmse_r) / 100000)
    rrr = float(average(accuracy))
    rmse_re = "RMSE:" + str(math.sqrt(rrr))
    print(time111, coverage, rmse_re)
    abccc = [time111, coverage, rmse_re]
    f_csv.writerow(abccc)

# print('total user number:'+str(user_num))
# print('total item number:'+str(item_num))
# print('number of 5 rating:'+str(rating.count('5')),'number of 4 rating:'+str(rating.count('4')),'number of 3 rating:'+str(rating.count('3')),'number of 2 rating:'+str(rating.count('2')),'number of 1 rating:'+str(rating.count('1')))
# print('rating density'+str(rating_density))

f.close()
