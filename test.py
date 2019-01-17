f = open('scoreData.txt')
s = f.readlines()
f.close()
s = s[3:]
t = []
for ss in s:
    if "'penalty': 'l2'" in ss and "'tol': 0.0001" in ss:
        t.append(ss[:7])
for tt in t:
    print(tt)