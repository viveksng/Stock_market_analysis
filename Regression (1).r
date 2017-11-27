
library("AzureML")
ws <- workspace()
dat <- download.datasets(ws, "FinalD.csv")

names(dat) = c("Y","Dl","Nlp","Cp")
dat = data.frame(dat)
head(dat)

#print(dat$Y[38])

for(i in 1:length(dat$Y)) # {"sell":1,"Hold":2,"Buy":3}
    for(j in length(dat$Y))
        if(dat$Y[i] == "Sell")
           { dat$Y[i] = 1
        }else if(dat$Y[i] == "Hold")
            {dat$Y[i] = 2
        }else{
            dat$Y[i] = 3 
        }
            
for(i in 1:length(dat$Cp)) ## {"Cp$Profit": 1, "Cp$loss":-1,"Cp$cost":0}
    for(j in length(dat$Cp))
        if(dat$Cp[i] == "P")
           { dat$Cp[i] = 1
        }else if(dat$Cp[i] == "L")
            {dat$Cp[i] = -1
        }else{
            dat$Cp[i] = 0 
        }        

#print(dat$Y[38])
#ex_dat = dat

#print(dat$Cp[112])

mul = lm(Y ~ Nlp+Dl, data = dat)

print(mul)

install.packages("rpart")

library(rpart)

n <- nrow(dat)
dfs <- dat[sample(n),]

#dfs

train_indices <- 1:round(0.7 * n)
train <- dfs[train_indices, ]
test_indices <- (round(0.7 * n) + 1):n
test <- dfs[test_indices, ]

tree <- rpart(Y ~., train, method = "class", minsplit=2, minbucket=1)

library(RColorBrewer)
library(rpart.plot)
fancyRpartPlot(tree)

pred_test <- predict(tree, test, type = "class")
conf <- table(test$Y, pred_test)
acc <- sum(diag(conf))/sum(conf)

acc

install.packages("foreign")

library(foreign)

head(dat)

library(nnet)

#p = dat$Nlp[sample(nrow(dat), 50), ]
#q = dat[, sample(ncol(df), 1)
q = dat$Nlp

#p
# normal_q = (q-min(q))/(max(q)-min(q))
#print(normal_q)
summary(normal_q)
rnorm(normal_q,0.50,0.3)
test = multinom(Y ~ Dl + Nlp + Cp, data = dat)

summary(test)
test$softmax

predict(test,dat)#{"sell":1,"Hold":2,"Buy":3}

Z = summary(test)$coefficients/summary(test)$standard.errors

head(fitted(test))

dat = as.vector(dat)
head(dat)
#set.seed(50)
#dat1 = sample(dat,size = 50,replace = T)

dat1 = dat[sample(nrow(dat),50), ]
head(dat1)

head(dat1)

predicted = predict(test,dat1,type="probs")

head(predicted)#{"sell":1,"Hold":2,"Buy":3}

bpp = cbind(dat1,predicted)# {"Cp$Profit": 1, "Cp$loss":-1,"Cp$cost":0}

bpp


