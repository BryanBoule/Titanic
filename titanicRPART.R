setwd("C:/Users/Bryan/Desktop/Titanic")
#but : donner une prediction des données test

#Charger package
install.packages("randomForest")
install.packages('sandwich')
install.packages('party')
install.packages("rpart")
install.packages('rattle')
install.packages('rpart.plot')
install.packages('RColorBrewer')
install.packages("ggplot2")

library(import)
library(robustbase)
library(Amelia)
library(mice)
library(DMwR)
library(randomForest)
library(sandwich)
library(party)
library(rpart)
library(rattle)
library(rpart.plot)
library(RColorBrewer)
library(ggplot2)
library(caret)
library(corrplot)
library(dplyr)
library(gridExtra)

#-------------------------------------------------------------------
# --------------------------IMPORT DATAS----------------------------
# ------------------------------------------------------------------
titanic.train <- read.csv(file= "train.csv", header=T, stringsAsFactors = F)
titanic.test <- read.csv(file = "test.csv", header=T, stringsAsFactors = F)

#Visualize features
glimpse(titanic.train)
glimpse(titanic.test)

#-------------------------------------------------------------------
# --------------------------CLEANING DATAS--------------------------
# ------------------------------------------------------------------

  #Merging datasets
  titanic.test$IsTrainSet <- F
  titanic.train$IsTrainSet <- T

  #Add same feature as titanic.train
  titanic.test$Survived <- NA

  #fusion
  titanic.full <- rbind(titanic.train, titanic.test)
  
  #Affichage des NAs
  missmap(titanic.full[-1], col=c('grey', 'steelblue'), y.cex=0.5, x.cex=0.8)
  
  #nb de NA values dans chaque colonne
  na_count <- sort(sapply(titanic.full[,1:ncol(titanic.full)], function(y) sum(is.na(y))), decreasing = T)
  na_count

  #Est ce que tout le monde a embarqué ?
  titanic.full[titanic.full$Embarked=='', "Embarked"] <- 'S'

  #On convertit en factor les colonnes interessantes
  titanic.full$Pclass <- as.factor(titanic.full$Pclass)
  titanic.full$Sex <- as.factor(titanic.full$Sex)
  titanic.full$Embarked <- as.factor(titanic.full$Embarked)
  
  #On gère les titres de personnes
    titanic.full$Name <- as.character(titanic.full$Name)
    titanic.full$Title <- sapply(titanic.full$Name, FUN=function(x) {strsplit(x, split='[,.]')[[1]][2]})
    titanic.full$Title <- sub(' ', '', titanic.full$Title)
    table(titanic.full$Title)
  
    #On simplifie pour réduire la diversité des features
    titanic.full$Title[titanic.full$Title %in% c('Mme', 'Mlle')] <- 'Mlle'
    titanic.full$Title[titanic.full$Title %in% c('Capt', 'Don', 'Major', 'Sir')] <- 'Sir'
    titanic.full$Title[titanic.full$Title %in% c('Dona', 'Lady', 'the Countess', 'Jonkheer')] <- 'Lady'
  
    #On remet en factor (jai mis as.factor au lieu de factor)
    titanic.full$Title <- as.factor(titanic.full$Title)
  
  #Gestion de la taille de la famille
  titanic.full$FamilySize <- titanic.full$SibSp + titanic.full$Parch + 1
  
  #On utilise le nom de famille pour grouper
  titanic.full$Surname <- sapply(titanic.full$Name, FUN=function(x) {strsplit(x, split='[,.]')[[1]][1]})
  titanic.full$FamilyID <- paste(as.character(titanic.full$FamilySize), titanic.full$Surname, sep="")
  titanic.full$FamilyID[titanic.full$FamilySize <= 2] <- 'Small'
  famIDs <- data.frame(table(titanic.full$FamilyID))
  famIDs <- famIDs[famIDs$Freq <= 2,]
  titanic.full$FamilyID[titanic.full$FamilyID %in% famIDs$Var1] <- 'Small'
  titanic.full$FamilyID <- factor(titanic.full$FamilyID)
  
  #Predictions de fare pour les NA
  upper.whisker <- boxplot.stats(titanic.full$Fare)$stats[5]
  outlier.filter <- titanic.full$Fare < upper.whisker
  titanic.full[outlier.filter,]
  
  fare.equation = "Fare ~ Pclass + Sex + Age + SibSp + Parch + Embarked"
  
  fare.model <- lm(
    formula = fare.equation,
    data = titanic.full[outlier.filter,]
  )

  fare.row <- titanic.full[is.na(titanic.full$Fare), c("Pclass", "Sex","Embarked", "Age", "SibSp", "Parch")]  
  fare.predictions <- predict(fare.model, newdata = fare.row)
  titanic.full[is.na(titanic.full$Fare), "Fare"] <- fare.predictions
  
  #On remplace les valeurs manquantes de l'age des passagers
  AgePassengers <- rpart(Age ~ Pclass + Sex + SibSp + Parch + Fare + Embarked + Title + FamilySize,
                         data= titanic.full[!is.na( titanic.full$Age),], 
                         method="anova")
  titanic.full$Age[is.na( titanic.full$Age)] <- predict(AgePassengers, titanic.full[is.na(titanic.full$Age),])  
  
  #Simplifications
  p1 <- qplot(titanic.full$Fare)
  
  titanic.full$Fare2 <- '30'
  titanic.full$Fare2[titanic.full$Fare < 30 & titanic.full$Fare >= 20] <- '20-30'
  titanic.full$Fare2[titanic.full$Fare < 20 & titanic.full$Fare >= 10] <- '10-20'
  titanic.full$Fare2[titanic.full$Fare < 10] <- '<10'
  aggregate(Survived ~ Fare2 + Pclass + Sex, data=titanic.full, FUN=function(x) {sum(x)/length(x)})
  
  p2 <- qplot(titanic.full$Fare2)
  
  grid.arrange(p1, p2, ncol=2, nrow = 1)
  
  titanic.full$Fare2 <- as.factor(titanic.full$Fare2)
#Nettoyage terminé
 
#On resplit 
titanic.test <- titanic.full[titanic.full$IsTrainSet == F,]
titanic.train <- titanic.full[titanic.full$IsTrainSet == T,]

titanic.train$Survived <- as.factor(titanic.train$Survived)

#modelisation + appr
fitControl <- trainControl(## 10-fold CV
  method = "repeatedcv",
  number = 10,
  verboseIter = T,
  ## repeated ten times
  repeats = 5)

#random forest___________________________
survived.equation <- "Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare2 + Embarked + Title + FamilySize + FamilyID"
survived.formula <- as.formula(survived.equation)

set.seed(415)

#titanic.model <- randomForest(formula = survived.formula, data=titanic.train, importance = TRUE, ntree=2000, mtry=3, nodesize=0.01*nrow(titanic.test))
titanic.model <- cforest(survived.formula, data = titanic.train, controls=cforest_unbiased(ntree=500, mtry=3))

install.packages('e1071', dependencies=TRUE)
fitrf <- train(survived.formula, 
               data = titanic.train, 
               method = "cforest", 
               trControl = fitControl,
               na.action=na.pass,
               tuneGrid = expand.grid(mtry=c(500)))

#Prediction
Survived <- predict(fitrf, titanic.test)
PassengerId <- titanic.test$PassengerId

#Mettre au format attendu
output.df <- as.data.frame(PassengerId)
output.df$Survived <- Survived

#Exporter les données
write.csv(output.df, file = "kaggle_submission.csv", row.names = F)

