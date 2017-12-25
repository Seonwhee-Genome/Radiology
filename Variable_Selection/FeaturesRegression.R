library(dplyr)
library(data.table)
FeatureData <- read.csv(file = "/home/seonwhee/Deep_Learning/MRImage_Pipeline/Variable_Selection/RadiomicFeatures.csv", header = TRUE, sep="\t", stringsAsFactors = FALSE)
## Removing general info, Feature Categories
FeatureData <- FeatureData[-1:-9,-2:-4]
FeatureData <- lapply(FeatureData, as.double) # change factors to doubles
FeatureData <- as.data.frame(FeatureData)  # change list to data.frame
FeatureData <- FeatureData[complete.cases(FeatureData),]  # remove NAs
Features <- FeatureData[,-1]

Response <- data.table(alive = c(0.0, 0.0, 1.0, 1.0, 0.0))   # y vector : 0 - Dead   1 - Alive
Features <- cbind(transpose(Features), Response)

##########  Linear Regression ###############
# Without penalty
OLS <- lm(alive ~. , data=Features)

########## LASSO #####################
library(lars)
# library(glmnet)
y <- Features$alive
x <- Features[,-1]
n <- nrow(x)
x <- matrix(unlist(x), nrow = n)  # change list to matrix

library(ncvreg)
set.seed(111)
cv_obj <- cv.ncvreg(x, y, penalty="lasso") # Cross-validation
plot(cv_obj)
optimal_lambda <- cv_obj$lambda.min
abline(v = log(optimal_lambda), col=4)

LASSO_ncvreg <- ncvreg(x,y, penalty = "lasso", lambda = optimal_lambda) # Fit
Estimation_ncvreg <- coef(LASSO_ncvreg)

library(lsgl)
LASSO_lsgl <- lsgl(x,y, lambda=optimal_lambda)
