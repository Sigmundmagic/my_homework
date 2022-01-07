read.csv("/Expressions.csv")
df <- read.csv("/Expressions.csv")
colnames(df)[1] <- "gene_id"
hist(df$Wild.type.5.s1)
plot(density(df$Wild.type.5.s1))
library(ggplot2)
library(tidyr)
df_long <- gather(df,"sample","expression",-gene_id)
ggplot(df_long,aes(expression)) +
  geom_density(aes(color = sample))


plot(df$Wild.type.5.s1,df$Wild.type.5.s2)
cor_test <- cor.test(df$Wild.type.5.s1,df$Wild.type.5.s2) # cor example test between 2 variables

cor_test
str(cor_test)
# pairs(df)

cor_test$estimate 
cor_test$p.value

library(Hmisc) # libraty with matrix_correlations

df_matrix <- as.matrix(df[-1]) # turn df into a matrix
dim(df_matrix)
df_matrix[1:5, 1:5]
apply(df_matrix, 2, typeof)


corr_matrix <- rcorr(df_matrix) # all correlations for pairs
str(corr_matrix)
corr_matrix

#View(corr_matrix$r) 
corr_matrix$P
corr_matrix$n

library(corrplot) # lib for cor visualization
corrplot(corr_matrix$r)

qplot(corr_matrix$r[upper.tri(corr_matrix$r)])+
  labs(x = 'cor coefficient', y = 'number sample pairs') #hist with cor. coef.

library(gplots)
# balloonplot(corr_matrix$r)
library(ggpubr)
scales::show_col(colors()[grepl('blue', colors())]) 

ggballoonplot(corr_matrix$r, fill = 'steelblue3')+
  labs(title = 'Correlation matrix', size = 'cor coef')
  
?ggballoonplot
??ggballoonplot

## PCA

library(factoextra)
library(FactoMineR)

df_t <- t(df[-1])
df_t[1:5, 1:5]
df[1:5, 1:5]
colnames(df_t) <- df$gene_id
pca_result <- PCA(df[-1], graph = T, ncp = 10)

print(pca_result)

eig.val <- get_eigenvalue(pca_result)
eig.val

fviz_eig(pca_result, addlabels = TRUE, ylim = c(0, 100))


var <- get_pca_var(pca_result)
var
var$contrib

fviz_pca_var(pca_result, col.var = "black")
var$coord
fviz_pca_var(pca_result, col.var = "black", axes = c(2, 3))

pca_coord <- as.data.frame(var$coord)
pca_coord$type <- ifelse(grepl('Wild.type', rownames(pca_coord)), 'wild', 'myc')
pca_coord$age <- ifelse(grepl('.24.', rownames(pca_coord)), 24, 5)

ggplot(pca_coord, aes(Dim.1, Dim.2))+
  geom_point(size = 5, aes(colour = factor(age), shape = type))+
  geom_text(aes(label = rownames(pca_coord)), vjust = -1)+
  labs(shape = 'type', colour = 'age', title = '')


corrplot(var$cos2, is.corr=FALSE)
ggballoonplot(var$cos2[, -1], fill = 'steelblue3')+
  labs(title = 'Correlation matrix', size = 'cor coef')
###### Spearman corr
spear <- cor.test(df$Wild.type.5.s1, df$Wild.type.5.s2, method = "spearman")
spear
pear <- cor.test(df$Wild.type.5.s1, df$Wild.type.5.s2, method = "spearman", exact =FALSE, alternative = "two.sided") # Spearman corr between two samples
Spearman_matrix <- Hmisc::rcorr(as.matrix(df_matrix), type = c("spearman"))
Spearman_matrix
library(reshape2)  # Heatmap for Spearman cor. matrix
corr_long <- melt(Spearman_matrix$r)
ggplot(corr_long, aes(Var1, Var2, fill = value))+
  geom_tile()+
  theme(axis.text.x = element_text(angle = 45, vjust = 0.5, hjust = 1))+
  labs(x = '', y = '', fill = 'corr coef')
######
library(reshape2)  # Heatmap for Pearson cor.
corr_long <- melt(corr_matrix$r)
ggplot(corr_long, aes(Var1, Var2, fill = value))+
  geom_tile()+
  theme(axis.text.x = element_text(angle = 45, vjust = 0.5, hjust = 1))+
  labs(x = '', y = '', fill = 'corr coef')
###### Diff. Gene expression
library("limma") 
library("edgeR") 
counts <- read.csv("C:/Users/Sergei/Documents/Expressions.csv", row.names = 1)
head(counts)
d0 <- DGEList(counts)
d0 <- calcNormFactors(d0)
d0
ls()
dim(d)
snames <- colnames(counts) 
snames
cultivar <- substr(snames, 1, nchar(snames) - 2) 
time <- substr(snames, nchar(snames) - 1, nchar(snames) - 1) 
time
cultivar
group <- interaction(cultivar, time) 
group
library(ggplot2)
d0_1 <- as.data.frame(d0)
plot_1 <- ggplot(d0_1, col = as.numeric(group))
plot_1
d0_1
mm <- model.matrix(~0 + group)
mm
y <- voom(d0_1, mm, plot = T)
y <- voom(counts$Wild.type.5.s1, counts$Wild.type.5.s2, mm, plot = TRUE)
y <- voom(counts$Wild.type.5.s1, counts$Wild.type.5.s2, d0_1, mm, plot = TRUE)
y <- voom(counts$Wild.type.5.s1, counts$Wild.type.5.s2, d0_1)
y <- voom(counts$Wild.type.5.s1, counts$Wild.type.5.s2)
y <- voom(counts$Wild.type.5.s1, counts$Wild.type.5.s2, mm)
y <- voom(d0_1)
y <- voom(d0_1, mm)
