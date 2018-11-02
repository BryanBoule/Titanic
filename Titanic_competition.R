#-------------------------------------------------------------------
# --------------------------TITANIC KAGGLE--------------------------
# ------------------------------------------------------------------

# Set path
getwd()
setwd("C:/Users/Bryan/Desktop/Titanic")

# Load Libraries
library(dplyr)
library(Amelia)
library(ggplot2)
library(caret)
library(rpart)
library(gridExtra) # grid.arrange

#-------------------------------------------------------------------
# --------------------------IMPORT DATAS----------------------------
# ------------------------------------------------------------------

  # Import Datas
  titanic.train <- read.csv(file= "train.csv", header=T, stringsAsFactors = F)
  titanic.test <- read.csv(file = "test.csv", header=T, stringsAsFactors = F)
  
  # Visualize features
  glimpse(titanic.train)
  glimpse(titanic.test)

#-------------------------------------------------------------------
# -----------------------DATA DESCRIPTION---------------------------
# ------------------------------------------------------------------
  
  #  Pclass      Passenger Class (1 = 1st; 2 = 2nd; 3 = 3rd)
  #  survival    Survival (0 = No; 1 = Yes)
  #  name        Name
  #  sex         Sex
  #  age         Age
  #  sibsp       Number of Siblings/Spouses Aboard 
    #              Siblings : Brother, Sister, Stepbrother, or Stepsister of Passenger Aboard Titanic
    #              Spouses  : Husband or Wife of Passenger Aboard Titanic (Mistresses and Fiances Ignored)
  #  parch       Number of Parents/Children Aboard
    #              Parents : Mother or Father of Passenger Aboard Titanic
    #              Child : Son, Daughter, Stepson, or Stepdaughter of Passenger Aboard Titanic  
  #  ticket      Ticket Number
  #  fare        Passenger Fare (British pound)
  #  cabin       Cabin
  #  embarked    Port of Embarkation (C = Cherbourg; Q = Queenstown; S = Southampton)
  #  boat        Lifeboat
  #  body Body   Identification Number
  #  home.dest   Home/Destination
  
#-------------------------------------------------------------------
# ----------------------------NA TREATMENT--------------------------
# ------------------------------------------------------------------

# ----------------------------CHOICE FOLLOWED-----------------------  
# Some feature engineering is done to treat input properly na values
# ------------------------------------------------------------------

  # Binary feature IsTrainSet to caracterize whether an observation is part of the trainset
  titanic.test$IsTrainSet <- F
  titanic.train$IsTrainSet <- T

  # Add Survived to-predict feature
  titanic.test$Survived <- NA
  
  # Merge train and test
  titanic.full <- rbind(titanic.train, titanic.test)
  
  # Display NA
  missmap(titanic.full[-1], col=c('grey', 'steelblue'), y.cex=0.5, x.cex=0.8)
  
  # Count NA for each feature
  na_count <- sort(sapply(titanic.full[,1:ncol(titanic.full)], function(y) sum(is.na(y))), decreasing = T)
  na_count  

  # Watch Embarked feature
  summary(titanic.full$Embarked)
  table(titanic.full$Embarked)
  
  # We need to get rid of ' ' observations, I chose to assigne ' ' to S 914/(2+270+123+914)% to be accurate
  titanic.full[titanic.full$Embarked=='', "Embarked"] <- 'S'
  
  # Convert into factor some features
  str(titanic.full)
  titanic.full$Pclass <- as.factor(titanic.full$Pclass)
  titanic.full$Sex <- as.factor(titanic.full$Sex)
  titanic.full$Embarked <- as.factor(titanic.full$Embarked)
  
  # Working on People's title
  head(titanic.full$Name)
  
  # I wish to have only the Mrs, Mr, ...
  titanic.full$Name <- as.character(titanic.full$Name)
  
  # We notice that '[,.]' are separators and the titles are always 2nd position
  titanic.full$Title <- sapply(titanic.full$Name, FUN=function(x) {strsplit(x, split='[,.]')[[1]][2]})
  
  # Remove the extra space
  titanic.full$Title <- sub(' ', '', titanic.full$Title)
  
  # Count titles
  table(titanic.full$Title)
  
  # Aggregations
  titanic.full$Title[titanic.full$Title %in% c('Mme', 'Mlle')] <- 'Mlle'
  titanic.full$Title[titanic.full$Title %in% c('Capt', 'Don', 'Major', 'Sir')] <- 'Sir'
  titanic.full$Title[titanic.full$Title %in% c('Dona', 'Lady', 'the Countess', 'Jonkheer')] <- 'Lady'
  table(titanic.full$Title)
  
  # Convert the char feature to factor
  titanic.full$Title <- as.factor(titanic.full$Title)
  
  # Managment of the family size
  titanic.full$FamilySize <- titanic.full$SibSp + titanic.full$Parch + 1
  sum(titanic.full$FamilySize)
  # It is the number of family number but no necessarily present on the boat
  
  sum(titanic.full$SibSp)
  
  # Aggregate according to family name (can introduce errors if two persons who don't know each other share the same surname)
  # Here, we use the first element
  titanic.full$Surname <- sapply(titanic.full$Name, FUN=function(x) {strsplit(x, split='[,.]')[[1]][1]})
  titanic.full$FamilyID <- paste(as.character(titanic.full$FamilySize), titanic.full$Surname, sep="")
  titanic.full$FamilyID[titanic.full$FamilySize <= 2] <- 'Small'
  famIDs <- data.frame(table(titanic.full$FamilyID))
  famIDs <- famIDs[famIDs$Freq <= 2,]
  titanic.full$FamilyID[titanic.full$FamilyID %in% famIDs$Var1] <- 'Small'
  titanic.full$FamilyID <- factor(titanic.full$FamilyID)
  
  # Predict the NA Fare
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
  
  # Age input for missing values
  AgePassengers <- rpart(Age ~ Pclass + Sex + SibSp + Parch + Fare + Embarked + Title + FamilySize,
                         data= titanic.full[!is.na( titanic.full$Age),], 
                         method="anova")
  titanic.full$Age[is.na( titanic.full$Age)] <- predict(AgePassengers, titanic.full[is.na(titanic.full$Age),])  
  
  # Aggregations on Fare
  # Split into 4 categories
  titanic.full$Fare2 <- '30'
  titanic.full$Fare2[titanic.full$Fare < 30 & titanic.full$Fare >= 20] <- '20-30'
  titanic.full$Fare2[titanic.full$Fare < 20 & titanic.full$Fare >= 10] <- '10-20'
  titanic.full$Fare2[titanic.full$Fare < 10] <- '<10'
  aggregate(Survived ~ Fare2 + Pclass + Sex, data=titanic.full, FUN=function(x) {sum(x)/length(x)})
  
  # Visualization of aggregations
  p1 <- qplot(titanic.full$Fare)
  p2 <- qplot(titanic.full$Fare2)
  grid.arrange(p1, p2, ncol=2, nrow = 1)
  
  # Convert to factor
  titanic.full$Fare2 <- as.factor(titanic.full$Fare2)
  
  # Cleaning and imputations over, let's check the number of NA : 418 expected 
  table(is.na(titanic.full))
    
  # Split the dataset for the modeling
  titanic.test.featured <- titanic.full[titanic.full$IsTrainSet == F,]
  titanic.train.featured <- titanic.full[titanic.full$IsTrainSet == T,]
  
  titanic.train.featured$Survived <- as.factor(titanic.train.featured$Survived)

#-------------------------------------------------------------------
# -------------------------------MODEL------------------------------
# ------------------------------------------------------------------

  # 10-fold CV
  fitControl <- trainControl(## 10-fold CV
    method = "repeatedcv",
    number = 10,
    verboseIter = T,
    ## repeated five times
    repeats = 5)
  
  # Write equation model
  survived.equation <- "Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare2 + Embarked + Title + FamilySize + FamilyID"
  survived.formula <- as.formula(survived.equation)
  
  set.seed(415)
  fitrf <- train(survived.formula, 
                 data = titanic.train.featured, 
                 method = "cforest", 
                 trControl = fitControl,
                 na.action = na.pass,
                 tuneGrid = expand.grid(mtry=c(500)))

#-------------------------------------------------------------------
# --------------------------PREDICTION------------------------------
# ------------------------------------------------------------------

  Survived <- predict(fitrf, titanic.test.featured)
  PassengerId <- titanic.test.featured$PassengerId

#-------------------------------------------------------------------
# -----------------------------RESULTS------------------------------
# ------------------------------------------------------------------

  # Create the result dataframe
  output.df <- as.data.frame(PassengerId)
  output.df$Survived <- Survived
  
  # Export the result dataframe
  write.csv(output.df, file = "kaggle_submission.csv", row.names = F)

#-------------------------------------------------------------------
# -----------------------------END----------------------------------
# ------------------------------------------------------------------

# Thank you for reading, do not hesitate to comment or contact me so that i can improve my results
# bryanboule@gmail.com