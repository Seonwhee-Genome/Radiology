library(dplyr)
library(data.table)
FeatureData <- read.csv(file = "/home/seonwhee/Deep_Learning/MRImage_Pipeline/Variable_Selection/RadiomicFeatures.csv", header = TRUE, sep="\t", stringsAsFactors = FALSE)
## Removing general info, Feature Categories
FeatureData <- FeatureData[-1:-9,-2:-4]
FeatureData <- FeatureData %>%
  lapply(as.double) %>% as.data.frame

FeatureData <- FeatureData[complete.cases(FeatureData),]  # remove NAs
Features <- FeatureData %>%
  select(-(X))

Response <- data.table(alive = c(0.0, 0.0, 1.0, 1.0, 0.0))   # y vector : 0 - Dead   1 - Alive
Features <- cbind(transpose(Features), Response)

##########  Linear Regression ###############
# Without penalty
OLS <- glm(alive ~. , data=Features)


library(lars)
# library(glmnet)
y <- Features$alive
x <- Features[,-1]
n <- nrow(x)
x <- matrix(unlist(x), nrow = n)  # change list to matrix

########## LASSO #####################
library(ncvreg)
set.seed(111)
cv_obj <- cv.ncvreg(x, y, penalty="lasso") # Cross-validation
optimal_lambda <- cv_obj$lambda.min
plot(cv_obj)
abline(v = log(optimal_lambda), col=4)

LASSO_ncvreg <- ncvreg(x,y, penalty = "lasso", lambda = optimal_lambda) # Fit
Estimation_ncvreg <- coef(LASSO_ncvreg)
Estimation_ncvreg

#library(lsgl)
#LASSO_lsgl <- lsgl(x,y, lambda=optimal_lambda)

###### Sure Independent Screening ######

library(SIS)
SIS_obj <- SIS(x, y, family = 'binomial', penalty = 'lasso')
SIS_obj$ix
SIS_obj$coef.est
