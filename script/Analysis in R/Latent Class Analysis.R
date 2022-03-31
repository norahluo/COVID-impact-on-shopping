# load data, combine demographic variables and other info
sacog_wDemo <- read.csv('data/rMoves_and_2020_matchLabel.csv', header = TRUE)
sacog_wDemo <- subset(sacog_wDemo, is.na(sacog_wDemo$match_id) == FALSE)
sacog <- readxl::read_excel('data/manipulatedData/sacog_with_new_variables.xlsx')
sacog_wDemo <- sacog_wDemo[c('match_id', 'hhveh', 'HouseholdIncome', 'PersonalIncome', 'Children.12To18', 'Children.5To12', 'Children.Under5', 'What.is.your.age.group.', 'Gender')]
colnames(sacog_wDemo) <- c('id', 'hhveh','hhIncome', 'pslIncome', 'child12to18', 'child5to12', 'childUnder5', 'age', 'gender')
sacog_demo <- merge(sacog, sacog_wDemo, by = 'id')

# fill in household income for single household
sacog_demo[sacog_demo$NumberInHousehold == 1,'hhIncome'] <- sacog_demo[sacog_demo$NumberInHousehold == 1,'pslIncome']

# create new var, indicate whether the household purchased  
cat <- c('PreparedFood', 'Groceries', 'OtherFood', 'PaperCleaning', 'Clothing', 'HomeOffice', 'Medication', 'ChildcareItems')
for (cat_ in cat) {
  sacog_demo[paste0('MayDidE', cat_)] <- ifelse(sacog_demo[paste0('May-Orders-',cat_)] > 0, 1, 2)
  sacog_demo[paste0('MayDid', cat_)] <- ifelse(sacog_demo[paste0('May-Trips-', cat_)] > 0, 1, 2)
}

library('poLCA')
f <- cbind(MayDidEPreparedFood, MayDidPreparedFood,
           MayDidEGroceries, MayDidGroceries,
           MayDidEOtherFood, MayDidOtherFood,
           MayDidEPaperCleaning, MayDidPaperCleaning,
           MayDidEClothing, MayDidClothing,
           MayDidEHomeOffice, MayDidHomeOffice,
           MayDidEMedication, MayDidMedication,
           MayDidEChildcareItems, MayDidChildcareItems) ~ 1

# when number of class equals to 4, bic is the minimum
#### didn't deal with the covariance matrix yet ####
bic_ <- rep(0, 10)
for (i in 2:10){
  lca <- poLCA(f, data = sacog_demo, nclass = i, nrep = 100, verbose = FALSE)
  bic_[i] <- lca$bic
}

lca <- poLCA(f, data = sacog_demo, nclass = 4, nrep = 100, verbose = FALSE)
sacog_demo$label <- lca$predclass

summary(sacog_demo[sacog_demo$label == 1, 'hhIncome'])

library('dplyr')
library('summarytools')
sacog_demo %>% group_by(label, hhIncome) %>% summarise(count=n())
librayr
  
  
  
