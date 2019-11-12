from a4_20824226 import *
b_,A_ = svm_read_problem(os.getcwd()+'/a9a')
b,A = shuffle(b_,A_)
train_data = A[int(0.05*len(A)):int(0.95*len(A))]
train_label = b[int(0.05*len(b)):int(0.95*len(b))]


test_data= A[:int(0.05*len(A))]+ A[int(0.95*len(A)):]

obj = MyMethod()
st = time.time()
obj.fit(train_data, train_label)
running_time = time.time() - st
predict_label = obj.predict(test_data)

print(predict_label)