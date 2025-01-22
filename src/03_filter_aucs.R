####################################################
#                      readme                      #  
####################################################

# export full list of clusters

####################################################
#                        env                       #  
####################################################

rm(list = ls())

library(dplyr)

####################################################
#                     variables                    #  
####################################################

auc.r <- data.table::fread("results/aucs-response.csv")
auc.i <- data.table::fread("results/aucs-irae.csv")

####################################################
#                       main                       #  
####################################################

# look at > 0.7 or > 0.65 and mean > 0.7; 0.3/0/35

auc.r$min <- pmin(auc.r$pitt, auc.r$nyc, auc.r$dls)
auc.r$mean <- rowMeans(auc.r[, c(2:4)])
auc.r$max <- pmax(auc.r$pitt, auc.r$nyc, auc.r$dls)

for.perm <- auc.r[(auc.r$min >=0.7 | 
               (auc.r$mean >= 0.7 & 
                  (auc.r$pitt >= 0.65 & auc.r$nyc >= 0.65 & auc.r$dls >= 0.65))) |
                 (auc.r$max <=0.3 | 
                    (auc.r$mean <= 0.3 & 
                       (auc.r$pitt <= 0.35 & auc.r$nyc <= 0.35 & auc.r$dls <= 0.35))), ]

write.csv(for.perm, file = "results/aucs_signif-response.csv",
          row.names = FALSE, quote = FALSE)

auc.r$min <- pmin(auc.r$pitt, auc.r$nyc)
auc.r$mean <- rowMeans(auc.r[, c(2:3)])
auc.r$max <- pmax(auc.r$pitt, auc.r$nyc)

for.mrs <- auc.r[(auc.r$min >=0.7 | 
                     (auc.r$mean >= 0.7 & 
                        (auc.r$pitt >= 0.65 & auc.r$nyc >= 0.65))) |
                    (auc.r$max <=0.3 | 
                       (auc.r$mean <= 0.3 & 
                          (auc.r$pitt <= 0.35 & auc.r$nyc <= 0.35))), ]
write.csv(for.mrs, file = "results/aucs_signif-mrs.csv",
          row.names = FALSE, quote = FALSE)



# for irae
# just look at pitt cohort
for.irae <- auc.i[auc.i$pitt >= 0.7 | auc.i$pitt <= 0.3, ]
write.csv(for.irae, file = "results/aucs_signif-irae.csv",
          row.names = FALSE, quote = FALSE)

