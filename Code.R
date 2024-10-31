#importing libraries

library(tidyr)
library(ggplot2)
library(caret)
library(car)
library(effects)
library(fastDummies)


# checking the data
str(lbw)
summary(lbw)

# preprocessing lbw data
# 01
lowbirthweight <- lbw$lbwR
birthratex <- lbw[,c(2,3,5:8)]
birthratexG <- gather(birthratex,"variable","value") 
ggplot(birthratexG,aes(value)) +
  facet_wrap(~ variable,scale="free_x") +
  geom_histogram()

lbwP <- preProcess(birthratex, method = c("BoxCox", "center", "scale"))
lbwxT <- predict(lbwP, birthratex)

lbwsxTG <- gather(lbwxT,"variable","value") 
ggplot(lbwsxTG,aes(value)) +
  facet_wrap(~ variable,scale="free_x") +
  geom_histogram()

#creating a new dataframe using tranformed predictors, untransformed factor, and response
GrowthR <- lbw$GrowthR
lbwTrans <- data.frame(lowbirthweight, lbwxT, GrowthR)

# creating a linear regression model
# 02
model1 <- lm(lowbirthweight ~ ., data = lbwTrans)
summary(model1)
plot(allEffects(model1))

# calculating VIF to check multicollinearity
# 03
vif(model1)

# creating auxiliary regression model using predictor with highest vif as a response
# 04
PropPov <- lbw$PropPov
auxmodel1x <- lbw[,c(2:5,7:8)]
auxmodel1xP <- preProcess(auxmodel1x, method = c("BoxCox", "center", "scale"))
auxmodel1xT <- predict(auxmodel1xP, auxmodel1x)
auxmodel1Trans <- data.frame(PropPov, auxmodel1xT)
auxmodel1 <- lm(PropPov ~ ., data = auxmodel1Trans)
summary(auxmodel1)
Anova(auxmodel1)

# creating second model by removing highest vif value predictor
# 05

model2x <- lbw[,c(2:5,7:8)]
model2xP <- preProcess(model2x, method = c("BoxCox", "center", "scale"))
model2xT <- predict(model2xP, model2x)
model2xTrans <- data.frame(lowbirthweight, model2xT)
model2 <- lm(lowbirthweight ~ ., data = model2xTrans)

vif(model2)

summary(model2)

# doing partial F-test to test null hypothesis
# 06
modelnull <- lm(lowbirthweight~1,data = model2xTrans)
anova(modelnull, model2)

#residual plots for model2
# 07
residualPlots(model2, pch = 16)

# performing score test on full model2 and each predictors
# 08
ncvTest(model2)
ncvTest(model2,~HInc)
ncvTest(model2,~PopDen)
ncvTest(model2,~GrowthR)
ncvTest(model2,~PropRent)
ncvTest(model2,~PropBlack)
ncvTest(model2,~PropHisp)

#influence, influence index, and dfbetas plot of model2
# 09
influencePlot(model2, id=list(n=5,col=2))
influenceIndexPlot(model2, id=list(n=5,col=2))
dfbetasPlots(model2,id.n=5)

# removing outliers and high leverage values
# 10
model3 <- lm(lowbirthweight ~ PopDen + PropRent + GrowthR + HInc + PropBlack + PropHisp,data = model2xTrans[-c(4,43,79,219),])
summary(model3)
Anova(model3)

residualPlots(model3,pch=16)

# creating new model including polynomial term for PropBlack and PropHisp using data from model2
# 11
model4 <- lm(lowbirthweight ~ PopDen+PropRent + GrowthR + HInc + 
               poly(PropBlack,2) + poly(PropHisp,2), data=model2xTrans)
summary(model4)
Anova(model4)

# dropping insignificant predictors
# 12
model5 <- lm(lowbirthweight ~ HInc + poly(PropBlack,2) + 
               poly(PropHisp, degree = 2), data=model2xTrans)
summary(model5)
Anova(model5)

plot(allEffects(model5))

# creating influence plot to see which observations stick out
# 13
influencePlot(model5)

# creating dfbetas plot
# 14 
dfbetasPlots(model5,id.n=5)

# creating new model by removing odd observation from model4
# 15
model6 <- lm(lowbirthweight ~ PopDen+PropRent + GrowthR + HInc + 
               poly(PropBlack,2) + poly(PropHisp,2), data=model2xTrans[-43,])
summary(model6)
Anova(model6)

# creating new model by removing odd observation from model5
# 16
model7 <- lm(lowbirthweight ~ HInc + poly(PropBlack,2) + 
               poly(PropHisp, degree = 2), data=model2xTrans[-43,])
summary(model7)
Anova(model7)

# running a new model on model7 by integrating significant predictor in model6 (PopDen)
# 17
model8 <- lm(lowbirthweight ~ PopDen + HInc + poly(PropBlack,2) + 
               poly(PropHisp, degree = 2), data=model2xTrans[-43,])
summary(model8)
Anova(model8)

# K-fold CV with 10 folds (model6, model7, model8)
# 18
set.seed(111)
myfolds <- createMultiFolds(lbw$lbwR[-43],10,10)
tc <- trainControl(method = "repeatedcv",
                   index = myfolds,
                   repeats=10,
                   verboseIter = TRUE)

model6CV <- train(lbwR ~ poly(PropBlack,2)+PopDen+PropRent+
                    GrowthR + HInc + poly(PropHisp,2), data=lbw[-43,],
                  method="lm",
                  preProcess=c("BoxCox","center","scale"),
                  trControl=tc)
model7CV <- train(lbwR ~ poly(PropBlack,2) +
                    + HInc + poly(PropHisp,2), data=lbw[-c(43),],
                  method="lm",
                  preProcess=c("BoxCox","center","scale"),
                  trControl=tc)
model8CV <- train(lbwR ~ poly(PropBlack,2) + PopDen
                  + HInc + poly(PropHisp,2), data=lbw[-c(43),],
                  method="lm",
                  preProcess=c("BoxCox","center","scale"),
                  trControl=tc)
model6CV
model7CV
model8CV
