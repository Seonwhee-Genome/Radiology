library(SIS)

set.seed(1)
# test if there is at least one argument: if not, return an error
if (length(args)==0) {
  stop("At least one argument must be supplied (input file).n", call.=FALSE)
} else if (length(args)==1) {
  # default output file
  args[2] = "out.txt"
}
n <- args[1]
p <- args[2]
x <- matrix(rnorm(n * p), n, p)

beta <- c(4,4,4,-6*sqrt(2),4/3)
eps <- rnorm(n)
y <- x[, 1:5] %*% beta + eps

# SIS
sis.obj <- SIS(x, y)
hat.S <- sis.obj$ix
hat.beta <- sis.obj$coef.est