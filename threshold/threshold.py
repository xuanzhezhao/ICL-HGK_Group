import numpy as np
from sklearn import preprocessing, neighbors, model_selection, svm
import pandas as pd
import pickle
import serial
import re
import random
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

df_A = pd.read_table('letter/a.txt',header=None,sep=',')
df_A = df_A.drop(df_A.columns[-1],axis=1)
A = np.array(df_A)
A=A.reshape((len(A),60,6))

for i in range(len(A)):
    temp = 0
    for j in range(len(A[0])):
        thres = 0.01
        if (abs(A[i][j][0]-A[i][j-1][0])<thres and abs(A[i][j][0]-A[i][j-2][0])<thres):
            temp = j-2
            break
    if(temp !=0):
        A[i][temp:][0:]=0.0;

A = A.reshape(len(A),len(A[0])*6)
df_A = pd.DataFrame(A)


df_B = pd.read_table('letter/b.txt',header=None,sep=',')
df_B = df_B.drop(df_B.columns[-1],axis=1)
B = np.array(df_B)
B=B.reshape((len(B),60,6))

for i in range(len(B)):
    temp = 0
    for j in range(len(B[0])):
        thres = 0.01
        if (abs(B[i][j][0]-B[i][j-1][0])<thres and abs(B[i][j][0]-B[i][j-2][0])<thres):
            temp = j-2
            break
    if(temp !=0):
        B[i][temp:][0:]=0.0;

B = B.reshape(len(B),len(B[0])*6)
df_B = pd.DataFrame(B)


df_C = pd.read_table('letter/c.txt',header=None,sep=',')
df_C = df_C.drop(df_C.columns[-1],axis=1)
C = np.array(df_C)
C=C.reshape((len(C),60,6))

for i in range(len(C)):
    temp = 0
    for j in range(len(C[0])):
        thres = 0.01
        if (abs(C[i][j][0]-C[i][j-1][0])<thres and abs(C[i][j][0]-C[i][j-2][0])<thres):
            temp = j-2
            break
    if(temp !=0):
        C[i][temp:][0:]=0.0;

C = C.reshape(len(C),len(C[0])*6)
df_C = pd.DataFrame(C)


df_D = pd.read_table('letter/d.txt',header=None,sep=',')
df_D = df_D.drop(df_D.columns[-1],axis=1)
D = np.array(df_D)
D=D.reshape((len(D),60,6))

for i in range(len(D)):
    temp = 0
    for j in range(len(D[0])):
        thres = 0.01
        if (abs(D[i][j][0]-D[i][j-1][0])<thres and abs(D[i][j][0]-D[i][j-2][0])<thres):
            temp = j-2
            break
    if(temp !=0):
        D[i][temp:][0:]=0.0;

D = D.reshape(len(D),len(D[0])*6)
df_D = pd.DataFrame(D)

df_E = pd.read_table('letter/e.txt',header=None,sep=',')
df_E = df_E.drop(df_E.columns[-1],axis=1)
E = np.array(df_E)
E=E.reshape((len(E),60,6))

for i in range(len(E)):
    temp = 0
    for j in range(len(E[0])):
        thres = 0.01
        if (abs(E[i][j][0]-E[i][j-1][0])<thres and abs(E[i][j][0]-E[i][j-2][0])<thres):
            temp = j-2
            break
    if(temp !=0):
        E[i][temp:][0:]=0.0;

E = E.reshape(len(E),len(E[0])*6)
df_E = pd.DataFrame(E)


df_F = pd.read_table('letter/f.txt',header=None,sep=',')
df_F = df_F.drop(df_F.columns[-1],axis=1)
F = np.array(df_F)
F=F.reshape((len(F),60,6))

for i in range(len(F)):
    temp = 0
    for j in range(len(F[0])):
        thres = 0.01
        if (abs(F[i][j][0]-F[i][j-1][0])<thres and abs(F[i][j][0]-F[i][j-2][0])<thres):
            temp = j-2
            break
    if(temp !=0):
        F[i][temp:][0:]=0.0;

F = F.reshape(len(F),len(F[0])*6)
df_F = pd.DataFrame(F)


df_G = pd.read_table('letter/g.txt',header=None,sep=',')
df_G = df_G.drop(df_G.columns[-1],axis=1)
G = np.array(df_G)
G=G.reshape((len(G),60,6))

for i in range(len(G)):
    temp = 0
    for j in range(len(G[0])):
        thres = 0.01
        if (abs(G[i][j][0]-G[i][j-1][0])<thres and abs(G[i][j][0]-G[i][j-2][0])<thres):
            temp = j-2
            break
    if(temp !=0):
        G[i][temp:][0:]=0.0;

G = G.reshape(len(G),len(G[0])*6)
df_G = pd.DataFrame(G)

df_H = pd.read_table('letter/h.txt',header=None,sep=',')
df_H = df_H.drop(df_H.columns[-1],axis=1)
H = np.array(df_H)
H=H.reshape((len(H),60,6))

for i in range(len(H)):
    temp = 0
    for j in range(len(H[0])):
        thres = 0.01
        if (abs(H[i][j][0]-H[i][j-1][0])<thres and abs(H[i][j][0]-H[i][j-2][0])<thres):
            temp = j-2
            break
    if(temp !=0):
        H[i][temp:][0:]=0.0;

H = H.reshape(len(H),len(H[0])*6)
df_H = pd.DataFrame(H)


df_I = pd.read_table('letter/i.txt',header=None,sep=',')
df_I = df_I.drop(df_I.columns[-1],axis=1)
I = np.array(df_I)
I=I.reshape((len(I),60,6))

for i in range(len(I)):
    temp = 0
    for j in range(len(I[0])):
        thres = 0.01
        if (abs(I[i][j][0]-I[i][j-1][0])<thres and abs(I[i][j][0]-I[i][j-2][0])<thres):
            temp = j-2
            break
    if(temp !=0):
        I[i][temp:][0:]=0.0;

I = I.reshape(len(I),len(I[0])*6)
df_I = pd.DataFrame(I)


df_J = pd.read_table('letter/j.txt',header=None,sep=',')
df_J = df_J.drop(df_J.columns[-1],axis=1)
J = np.array(df_J)
J=J.reshape((len(J),60,6))

for i in range(len(J)):
    temp = 0
    for j in range(len(J[0])):
        thres = 0.01
        if (abs(J[i][j][0]-J[i][j-1][0])<thres and abs(J[i][j][0]-J[i][j-2][0])<thres):
            temp = j-2
            break
    if(temp !=0):
        J[i][temp:][0:]=0.0;

J = J.reshape(len(J),len(J[0])*6)
df_J = pd.DataFrame(J)


df_K = pd.read_table('letter/k.txt',header=None,sep=',')
df_K = df_K.drop(df_K.columns[-1],axis=1)
K = np.array(df_K)
K=K.reshape((len(K),60,6))

for i in range(len(K)):
    temp = 0
    for j in range(len(K[0])):
        thres = 0.01
        if (abs(K[i][j][0]-K[i][j-1][0])<thres and abs(K[i][j][0]-K[i][j-2][0])<thres):
            temp = j-2
            break
    if(temp !=0):
        K[i][temp:][0:]=0.0;

K = K.reshape(len(K),len(K[0])*6)
df_K = pd.DataFrame(K)


df_L = pd.read_table('letter/l.txt',header=None,sep=',')
df_L = df_L.drop(df_L.columns[-1],axis=1)
L = np.array(df_L)
L=L.reshape((len(L),60,6))

for i in range(len(L)):
    temp = 0
    for j in range(len(L[0])):
        thres = 0.01
        if (abs(L[i][j][0]-L[i][j-1][0])<thres and abs(L[i][j][0]-L[i][j-2][0])<thres):
            temp = j-2
            break
    if(temp !=0):
        L[i][temp:][0:]=0.0;

L = L.reshape(len(L),len(L[0])*6)
df_L = pd.DataFrame(L)


df_M = pd.read_table('letter/m.txt',header=None,sep=',')
df_M = df_M.drop(df_M.columns[-1],axis=1)
M = np.array(df_M)
M=M.reshape((len(M),60,6))

for i in range(len(M)):
    temp = 0
    for j in range(len(M[0])):
        thres = 0.01
        if (abs(M[i][j][0]-M[i][j-1][0])<thres and abs(M[i][j][0]-M[i][j-2][0])<thres):
            temp = j-2
            break
    if(temp !=0):
        M[i][temp:][0:]=0.0;

M = M.reshape(len(M),len(M[0])*6)
df_M = pd.DataFrame(M)


df_N = pd.read_table('letter/n.txt',header=None,sep=',')
df_N = df_N.drop(df_N.columns[-1],axis=1)
N = np.array(df_N)
N=N.reshape((len(N),60,6))

for i in range(len(N)):
    temp = 0
    for j in range(len(N[0])):
        thres = 0.01
        if (abs(N[i][j][0]-N[i][j-1][0])<thres and abs(N[i][j][0]-N[i][j-2][0])<thres):
            temp = j-2
            break
    if(temp !=0):
        N[i][temp:][0:]=0.0;

N = N.reshape(len(N),len(N[0])*6)
df_N = pd.DataFrame(N)


df_O = pd.read_table('letter/o.txt',header=None,sep=',')
df_O = df_O.drop(df_O.columns[-1],axis=1)
O = np.array(df_O)
O=O.reshape((len(O),60,6))

for i in range(len(O)):
    temp = 0
    for j in range(len(O[0])):
        thres = 0.01
        if (abs(O[i][j][0]-O[i][j-1][0])<thres and abs(O[i][j][0]-O[i][j-2][0])<thres):
            temp = j-2
            break
    if(temp !=0):
        O[i][temp:][0:]=0.0;

O = O.reshape(len(O),len(O[0])*6)
df_O = pd.DataFrame(O)


df_P = pd.read_table('letter/p.txt',header=None,sep=',')
df_P = df_P.drop(df_P.columns[-1],axis=1)
P = np.array(df_P)
P=P.reshape((len(P),60,6))

for i in range(len(P)):
    temp = 0
    for j in range(len(P[0])):
        thres = 0.01
        if (abs(P[i][j][0]-P[i][j-1][0])<thres and abs(P[i][j][0]-P[i][j-2][0])<thres):
            temp = j-2
            break
    if(temp !=0):
        P[i][temp:][0:]=0.0;

P = P.reshape(len(P),len(P[0])*6)
df_P = pd.DataFrame(P)


df_Q = pd.read_table('letter/q.txt',header=None,sep=',')
df_Q = df_Q.drop(df_Q.columns[-1],axis=1)
Q = np.array(df_Q)
Q=Q.reshape((len(Q),60,6))

for i in range(len(Q)):
    temp = 0
    for j in range(len(Q[0])):
        thres = 0.01
        if (abs(Q[i][j][0]-Q[i][j-1][0])<thres and abs(Q[i][j][0]-Q[i][j-2][0])<thres):
            temp = j-2
            break
    if(temp !=0):
        Q[i][temp:][0:]=0.0;

Q = Q.reshape(len(Q),len(Q[0])*6)
df_Q = pd.DataFrame(Q)


df_R = pd.read_table('letter/r.txt',header=None,sep=',')
df_R = df_R.drop(df_R.columns[-1],axis=1)
R = np.array(df_R)
R=R.reshape((len(R),60,6))

for i in range(len(R)):
    temp = 0
    for j in range(len(R[0])):
        thres = 0.01
        if (abs(R[i][j][0]-R[i][j-1][0])<thres and abs(R[i][j][0]-R[i][j-2][0])<thres):
            temp = j-2
            break
    if(temp !=0):
        R[i][temp:][0:]=0.0;

R = R.reshape(len(R),len(R[0])*6)
df_R = pd.DataFrame(R)


df_S = pd.read_table('letter/s.txt',header=None,sep=',')
df_S = df_S.drop(df_S.columns[-1],axis=1)
S = np.array(df_S)
S=S.reshape((len(S),60,6))

for i in range(len(S)):
    temp = 0
    for j in range(len(S[0])):
        thres = 0.01
        if (abs(S[i][j][0]-S[i][j-1][0])<thres and abs(S[i][j][0]-S[i][j-2][0])<thres):
            temp = j-2
            break
    if(temp !=0):
        S[i][temp:][0:]=0.0;

S = S.reshape(len(S),len(S[0])*6)
df_S = pd.DataFrame(S)


df_T = pd.read_table('letter/t.txt',header=None,sep=',')
df_T = df_T.drop(df_T.columns[-1],axis=1)
T = np.array(df_T)
T=T.reshape((len(T),60,6))

for i in range(len(T)):
    temp = 0
    for j in range(len(T[0])):
        thres = 0.01
        if (abs(T[i][j][0]-T[i][j-1][0])<thres and abs(T[i][j][0]-T[i][j-2][0])<thres):
            temp = j-2
            break
    if(temp !=0):
        T[i][temp:][0:]=0.0;

T = T.reshape(len(T),len(T[0])*6)
df_T = pd.DataFrame(T)


df_U = pd.read_table('letter/u.txt',header=None,sep=',')
df_U = df_U.drop(df_U.columns[-1],axis=1)
U = np.array(df_U)
U=U.reshape((len(U),60,6))

for i in range(len(U)):
    temp = 0
    for j in range(len(U[0])):
        thres = 0.01
        if (abs(U[i][j][0]-U[i][j-1][0])<thres and abs(U[i][j][0]-U[i][j-2][0])<thres):
            temp = j-2
            break
    if(temp !=0):
        U[i][temp:][0:]=0.0;

U = U.reshape(len(U),len(U[0])*6)
df_U = pd.DataFrame(U)


df_V= pd.read_table('letter/v.txt',header=None,sep=',')
df_V = df_V.drop(df_V.columns[-1],axis=1)
V = np.array(df_V)
V=V.reshape((len(V),60,6))

for i in range(len(V)):
    temp = 0
    for j in range(len(V[0])):
        thres = 0.01
        if (abs(V[i][j][0]-V[i][j-1][0])<thres and abs(V[i][j][0]-V[i][j-2][0])<thres):
            temp = j-2
            break
    if(temp !=0):
        V[i][temp:][0:]=0.0;

V = V.reshape(len(V),len(V[0])*6)
df_V = pd.DataFrame(V)


df_W = pd.read_table('letter/w.txt',header=None,sep=',')
df_W = df_W.drop(df_W.columns[-1],axis=1)
W = np.array(df_W)
W=W.reshape((len(W),60,6))

for i in range(len(W)):
    temp = 0
    for j in range(len(W[0])):
        thres = 0.01
        if (abs(W[i][j][0]-W[i][j-1][0])<thres and abs(W[i][j][0]-W[i][j-2][0])<thres):
            temp = j-2
            break
    if(temp !=0):
        W[i][temp:][0:]=0.0;

W = W.reshape(len(W),len(W[0])*6)
df_W = pd.DataFrame(W)


df_X = pd.read_table('letter/x.txt',header=None,sep=',')
df_X = df_X.drop(df_X.columns[-1],axis=1)
X = np.array(df_X)
X=X.reshape((len(X),60,6))

for i in range(len(X)):
    temp = 0
    for j in range(len(X[0])):
        thres = 0.01
        if (abs(X[i][j][0]-X[i][j-1][0])<thres and abs(X[i][j][0]-X[i][j-2][0])<thres):
            temp = j-2
            break
    if(temp !=0):
        X[i][temp:][0:]=0.0;

X = X.reshape(len(X),len(X[0])*6)
df_X = pd.DataFrame(X)


df_Y = pd.read_table('letter/y.txt',header=None,sep=',')
df_Y = df_Y.drop(df_Y.columns[-1],axis=1)
Y = np.array(df_Y)
Y=Y.reshape((len(Y),60,6))

for i in range(len(Y)):
    temp = 0
    for j in range(len(Y[0])):
        thres = 0.01
        if (abs(Y[i][j][0]-Y[i][j-1][0])<thres and abs(Y[i][j][0]-Y[i][j-2][0])<thres):
            temp = j-2
            break
    if(temp !=0):
        Y[i][temp:][0:]=0.0;

Y = Y.reshape(len(Y),len(Y[0])*6)
df_Y = pd.DataFrame(Y)


df_Z = pd.read_table('letter/z.txt',header=None,sep=',')
df_Z = df_Z.drop(df_Z.columns[-1],axis=1)
Z = np.array(df_Z)
Z=Z.reshape((len(Z),60,6))

for i in range(len(Z)):
    temp = 0
    for j in range(len(Z[0])):
        thres = 0.01
        if (abs(Z[i][j][0]-Z[i][j-1][0])<thres and abs(Z[i][j][0]-Z[i][j-2][0])<thres):
            temp = j-2
            break
    if(temp !=0):
        Z[i][temp:][0:]=0.0;

Z = Z.reshape(len(Z),len(Z[0])*6)
df_Z = pd.DataFrame(Z)

df = df_A.append(df_B)
df = df.append(df_C)
df = df.append(df_D)
df = df.append(df_E)
df = df.append(df_F)
df = df.append(df_G)
df = df.append(df_H)
df = df.append(df_I)
df = df.append(df_J)
df = df.append(df_K)
df = df.append(df_L)
df = df.append(df_M)
df = df.append(df_N)
df = df.append(df_O)
df = df.append(df_P)
df = df.append(df_Q)
df = df.append(df_R)
df = df.append(df_S)
df = df.append(df_T)
df = df.append(df_U)
df = df.append(df_V)
df = df.append(df_W)
df = df.append(df_X)
df = df.append(df_Y)
df = df.append(df_Z)

print(df.shape)

class_a = [0 for i in range(len(A))]
class_b = [1 for i in range(len(B))]
class_c = [2 for i in range(len(C))]
class_d = [3 for i in range(len(D))]
class_e = [4 for i in range(len(E))]
class_f = [5 for i in range(len(F))]
class_g = [6 for i in range(len(G))]
class_h = [7 for i in range(len(H))]
class_i = [8 for i in range(len(I))]
class_j = [9 for i in range(len(J))]
class_k = [10 for i in range(len(K))]
class_l = [11 for i in range(len(L))]
class_m = [12 for i in range(len(M))]
class_n = [13 for i in range(len(N))]
class_o = [14 for i in range(len(O))]
class_p = [15 for i in range(len(P))]
class_q = [16 for i in range(len(Q))]
class_r = [17 for i in range(len(R))]
class_s = [18 for i in range(len(S))]
class_t = [19 for i in range(len(T))]
class_u = [20 for i in range(len(U))]
class_v = [21 for i in range(len(V))]
class_w = [22 for i in range(len(W))]
class_x = [23 for i in range(len(X))]
class_y = [24 for i in range(len(Y))]
class_z = [25 for i in range(len(Z))]

y_label = np.append(class_a,class_b)
y_label = np.append(y_label,class_c)
y_label = np.append(y_label,class_d)
y_label = np.append(y_label,class_e)
y_label = np.append(y_label,class_f)
y_label = np.append(y_label,class_g)
y_label = np.append(y_label,class_h)
y_label = np.append(y_label,class_i)
y_label = np.append(y_label,class_j)
y_label = np.append(y_label,class_k)
y_label = np.append(y_label,class_l)
y_label = np.append(y_label,class_m)
y_label = np.append(y_label,class_n)
y_label = np.append(y_label,class_o)
y_label = np.append(y_label,class_p)
y_label = np.append(y_label,class_q)
y_label = np.append(y_label,class_r)
y_label = np.append(y_label,class_s)
y_label = np.append(y_label,class_t)
y_label = np.append(y_label,class_u)
y_label = np.append(y_label,class_v)
y_label = np.append(y_label,class_w)
y_label = np.append(y_label,class_x)
y_label = np.append(y_label,class_y)
y_label = np.append(y_label,class_z)

y_label = y_label.reshape(2680)

X = np.array(df)
y = np.array(y_label)

with open('save_letter.pickle', 'wb') as f:
    pickle.dump([X,y], f)

