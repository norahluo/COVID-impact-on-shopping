sacog_wDemo <- read.csv('data/rMoves_and_2020_matchLabel.csv', header = TRUE)
sacog_wDemo <- subset(sacog_wDemo, is.na(sacog_wDemo$match_id) == FALSE)
sacog <- readxl::read_excel('data/manipulatedData/sacog_with_new_variables.xlsx')

sacog$delivery <- ifelse((sacog$`DoInStage23-ContactlessDoor` > 3) | (sacog$`DoInStage23-ContactlessCurbside` > 3) | (sacog$`DoInStage23-ContactlessDriverless` > 3) | (sacog$`DoInStage23-DroneDelivery` > 3), 1, 2)
sacog$pickup <- ifelse((sacog$`DoInStage23-CurbsidePickup` > 3) | (sacog$`DoInStage23-InStorePickup` > 3) | (sacog$`DoInStage23-LockerPickup` > 3), 1, 2)
sacog$visit <- ifelse((sacog$`DoInStage23-VisitBar` > 3) | (sacog$`DoInStage23-VisitRestaurant` > 3) | (sacog$`DoInStage23-VisitRetail` > 3), 1, 2)

intention <- cbind(delivery, pickup, visit) ~ 1

library(poLCA)
bic_ <- rep(0, 10)
for (i in 2: 10){
  lca <- poLCA(intention, data = sacog, nclass = i, nrep = 100, verbose = FALSE)
  bic_[i] <- lca$bic
}
bic_
lca <- poLCA(intention, data = sacog, nclass = 3, nrep = 100, verbose = FALSE)
lca$probs$delivery[,'Pr(1)']
lca$predclass
lca$P
rbind('prob' = lca$P, 'delivery' = lca$probs$delivery[,'Pr(1)'], 'pickup' = lca$probs$pickup[,'Pr(1)'], 'visit' = lca$probs$visit[,'Pr(1)'])
write.csv(lca$predclass, 'output/IntentionClass.csv')
