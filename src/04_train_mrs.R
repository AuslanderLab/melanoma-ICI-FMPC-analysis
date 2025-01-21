####################################################
#                      readme                      #  
####################################################

# compute risk score for clinical response

####################################################
#                        env                       #  
####################################################

rm(list = ls())

library(dplyr)

# accesses tidyr namespace

####################################################
#                     functions                    #  
####################################################

# just puts a regular format with samples as rows
# and features as columns
reformat_data <- function(mat, cln){
  from <- names(mat[, -1])[1]
  to <- names(mat)[ncol(mat)]
  
  df <- mat %>%
    tidyr::pivot_longer(!!sym(from):!!sym(to)) %>%
    tidyr::pivot_wider(names_from = V1,
                       values_from = value) %>%
    inner_join(cln %>%
                 select(Sample, Response),
               by = c("name" = "Sample"))
  
  names(df)[2:(length(names(df))-1)] <- paste0("C", names(df)[2:(length(names(df))-1)])
  
  return(df)
}


####################################################
#                    variables                     #  
####################################################

cln = as.data.frame(data.table::fread("supp_dataset_1.csv"), header = TRUE))
cln$Response <- ifelse(cln$Study_Clin_Response == "Responder", 0, 1)
cln$BMI <- as.numeric(cln$BMI)
cln$pesc <- ifelse(cln$Timepoint_days <= 120, 1, 0)
pitt = as.data.frame(data.table::fread('supp_dataset_4-$PITT.csv'))

s1 = intersect(cln$Sample, names(pitt)) 

clst <- as.data.frame(data.table::fread('results/aucs_signif-irae.csv', header = TRUE))$V1

pitt.df <- pitt[pitt$V1 %in% clst, ]

####################################################
#                       main                       #  
####################################################

rowSums(pitt.df[, -1] == 0) > length(s1)-((length(s1)*0.05))

pitt.mat <- reformat_data(pitt.df, cln)

### regression model for MRS i.e. get the betas.
covs <- c("Abx_Use", "Sex", "BMI", "pesc",
          "PPI_use_at_time_of_collection")

cln.cov <- cln[, c("Sample", covs)]
df.test <- merge(pitt.mat, cln.cov, by.x = "name", by.y = "Sample")
df.test$Abx_Use[df.test$Abx_Use == "Used; also had MTX use"] <- "Used"

# check if we just do prescence/absence because this is how the scores will be computed
df.med <- data.frame(clst = names(pitt.mat[, 2:(ncol(pitt.mat)-1)]),
                     med = unlist(lapply(pitt.mat[, 2:(ncol(pitt.mat)-1)], median)))

df.bin <- pitt.mat %>%
  tidyr::pivot_longer(C142:C18821,
                      names_to = "clst") %>%
  select(name, clst, value) %>%
  inner_join(df.med, by = "clst") %>%
  mutate(seq.bin = ifelse(value >= med, 1, 0)) %>%
  select(name, clst, seq.bin) %>%
  tidyr::pivot_wider(names_from = clst, 
                     values_from = seq.bin) %>%
  left_join(cln[, c("Sample", "Response", covs)], 
             by = c("name" = "Sample")) 

cd.res.log <- data.frame()
for(i in paste0("C", clst)){
  
  for(j in covs){
    coef <- summary(glm(formula = df.bin[[i]] ~ df.bin[[j]]),
                    family = "binomial")$coefficients
    cd.res.log <- rbind(cd.res.log, 
                        data.frame("cov" = j,
                                   "clst" = i,
                                   "p" = coef[2, 4]))
  }
  
}

lapply(cd.res.log %>% tidyr::pivot_wider(names_from = cov,
                                         values_from = p), summary)


# base on this just PPI 

# now just calculate using anything that should be covariate adjusted
cd.betas.adj <- data.frame()
for(i in paste0("C", clst)){
  coef <- summary(glm(formula = as.factor(Response) ~ df.bin[[i]] + PPI_use_at_time_of_collection,
                      data = df.bin, family = "binomial"))$coefficients
  cd.betas.adj <- rbind(cd.betas.adj, 
                        data.frame("clst" = i,
                                   "beta" = coef[2,1],
                                   "pvalue" = coef[2, 4]))
}

write.table(cd.betas.adj, file = ("results/betas-ppi_adj-pitt.tsv"),
            row.names = FALSE, sep = "\t", quote = FALSE)


# make the actual burden score now

# with betas
cd.mrs <- df.bin %>% 
  tidyr::pivot_longer(C142:C18821,
                      names_to = "clst") %>%
  inner_join(cd.betas.adj, by = "clst") %>%
  mutate(tot = value * beta) %>%
  select(name, Response, tot) %>%
  group_by(name, Response) %>%
  summarise(mrs = sum(tot)) %>%
  ungroup()


# try pruning & thresholding
cd.prt <- df.bin %>% 
  tidyr::pivot_longer(C142:C18821,
                      names_to = "clst") %>%
  inner_join(cd.betas.adj %>% filter(pvalue < 0.05),
             by = "clst") %>%
  mutate(tot = value * beta) %>%
  select(name, Response, tot) %>%
  group_by(name, Response) %>%
  summarise(mrs = sum(tot)) %>%
  ungroup()


cd.mrs$pruning.thresholding <- "no"
cd.prt$pruning.thresholding <- "0.05"

all.mrs <- rbind(cd.mrs, cd.prt)


write.table(cd.prt, row.names = FALSE, quote = FALSE, sep = "\t",
            file = "snapshots/corrected_normalization_june_2024/mrs-pitt.tsv")


