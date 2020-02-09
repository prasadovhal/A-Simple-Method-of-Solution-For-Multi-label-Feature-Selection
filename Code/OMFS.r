### mifs
eps = 0.01
EPS = 1e-3
library(NMF)
library(e1071)
#data = read.csv('emotions.csv',sep = ',') # dataset variables = 72, dataset lables = 6, dataset examples = 593
data = read.csv('scene.csv',sep = ',') # dataset variables = 294, dataset lables = 6 , dataset examples = 2407
#data = read.csv('yeast.csv',sep = ',') # dataset variables = 103, dataset lables = 14 , dataset examples = 2417

var_size = 294 # emotion = 72, scene = 294, yeast = 103

X = as.matrix(data[,1:var_size])
Y = as.matrix(data[,-(1:var_size)])
folds = 5

#parameters alpha,beta,gamma
alpha = 0.4 
beta  = 1
gamma = 0.8
O = 0
lamda_v = 1e-6
lamda_b = 1e-6
lamda_w = 1e-8

###MIFS STEPS
#1. Initialize W,V,B
ras <- nmf(Y,rank=4,method='lee')
V = basis(ras)
B = coef(ras)
W = matrix(1,ncol(X),ncol(V))


while(TRUE)
{
	D = 0
	for(i in 1:ncol(W))
	{
		D[i] = 1/(2 * sqrt(t(W[i,]) %*% W[i,] + eps )) 
	}
	D = diag(D)
	#cat('D:',D,'\n')

	dO_dw = 2 * ( t(X) %*% (X %*% W - V) + (gamma * W %*% D) ) 
	W = W - lamda_w * (dO_dw)


	## objective function is 
	Onew = norm((X %*% W - V),type=c("F","2"))  + 2 * gamma * sum(diag(W %*% D %*% t(W))) 
	print(Onew)

	### output  || W ||

		if (abs(O-Onew) < EPS)
			break
		else
			O = Onew
}


print(W)
W_norm = 0
for(i in 1:nrow(W))
{
	W_norm[i] = norm(W[i,],"2")
	cat('W_norm:of',i, 'attribute = ',W_norm[i],'\n')
}


#######################################################################################################

### w_norm in decreasing order
q = order(W_norm,decreasing=T)
newX = X[,q]
newData = cbind.data.frame(newX,Y)

#######################################################################################################

### with subsets
subset_mean_mic_avg = 0

for(m in seq(8,ncol(X),by=8))
{
	mic_avg = hm_loss = 0
	folds = sample(5,nrow(newData),replace=T)
	Data = newData[,c(1:m,(ncol(X)+1):ncol(data))]

	for(i in 1:folds)
	{
		test = Data[which(folds== i),]
		X_test = test[,(1:m)]
		Y_test = test[,-(1:m)]
		
		train = Data[-which(folds== i),]
		X_train = train[,(1:m)]
		Y_train = train[,-(1:m)]
		
		t = matrix(0,2,2)
		for(j in 1:ncol(Y))
		{
			model = svm(x = X_train, y = Y_train[,j],type="C",kernel='linear')
			pred = predict(model,X_test)
			t = t +  table(pred,Y_test[,j])
		}

		## CALCULATE MICRO AVRG
		mic_avg[i] = (2 * t[2,2]) / (2 * t[2,2] + t[2,1] + t[1,2])
		#cat('mic_avg = ',mic_avg,'\n')
		hm_loss[i] = (t[1,2] + t[2,1]) / sum(t)
		#cat('hamming loss = ',hm_loss,'\n')
	}

	subset_mean_mic_avg[(m/8)] = mean(mic_avg)
	#cat('mic_avg alll = ',mic_avg,'\n')
	cat('mean_mic_avg alllsubset ',(m/1) ,'=' ,mean(mic_avg),'\n')
	cat('mean_hamming loss alllsubset ',(m/1) ,'=' ,mean(hm_loss),'\n')
}

#plot(1:(m/10),subset_mean_mic_avg,type='b')

