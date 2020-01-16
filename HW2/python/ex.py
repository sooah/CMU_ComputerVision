import pickle
import matplotlib.pyplot as plt
import numpy as np

with open('rand_eu_acc_200_500.pkl', 'rb') as handle:
    rand_eu_acc = pickle.load(handle)
with open('rand_chi_acc_200_500.pkl', 'rb') as handle:
    rand_chi_acc = pickle.load(handle)
with open('harris_eu_acc_200_500.pkl', 'rb') as handle:
    harris_eu_acc = pickle.load(handle)
with open('harris_chi_acc_200_500.pkl', 'rb') as handle:
    harris_chi_acc = pickle.load(handle)

knn = 40

rand_eu = np.zeros((knn,1))
for k in range(1,knn+1):
    acc_result = rand_eu_acc[k-1,:].tolist()
    acc = sum(acc_result)/len(acc_result)
    rand_eu[k-1,0] = acc
print('rand euclidean accuracy')
print('%d th knn is best'%(np.argmax(rand_eu[:,0])+1))
plt.plot(list(range(1,knn+1)) , rand_eu.tolist())
plt.show()
plt.savefig('rand_euclidean_knn.jpeg')
plt.close()

rand_chi = np.zeros((knn,1))
for k in range(1,knn+1):
    acc_result = rand_chi_acc[k-1,:].tolist()
    acc = sum(acc_result)/len(acc_result)
    rand_chi[k-1,0] = acc
print('rand chi accuracy')
print('%d th knn is best'%(np.argmax(rand_chi[:,0])+1))
plt.plot(list(range(1,knn+1)), rand_chi.tolist())
plt.show()
plt.savefig('rand_chi_knn.jpeg')
plt.close()

harris_eu = np.zeros((knn,1))
for k in range(1,knn+1):
    acc_result = harris_eu_acc[k-1,:].tolist()
    acc = sum(acc_result)/len(acc_result)
    harris_eu[k-1,0] = acc
print('harris eu accuracy')
print('%d th knn is best'%(np.argmax(harris_eu[:,0])+1))
plt.plot(list(range(1,knn+1)), harris_eu.tolist())
plt.show()
plt.savefig('harris_eu_knn.jpeg')
plt.close()

harris_chi = np.zeros((knn,1))
for k in range(1,knn+1):
    acc_result = harris_chi_acc[k-1,:].tolist()
    acc = sum(acc_result)/len(acc_result)
    harris_chi[k-1,0] = acc
print('harris chi accuracy')
print('%d th knn is best'%(np.argmax(harris_chi[:,0])+1))
plt.plot(list(range(1,knn+1)), harris_chi.tolist())
plt.show()
plt.savefig('harris_chi_knn.jpeg')
plt.close()