####################################################
#                      readme                      #  
####################################################

# validate risk score for clinical response
# note that risk is actually "response"

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

cln = as.data.frame(data.table::fread('supp_dataset_1.csv', header = TRUE))
cln$Response <- ifelse(cln$Study_Clin_Response == "Responder", 0, 1)
ny = as.data.frame(data.table::fread("supp_dataset_4-$NY.csv"))

s2 = intersect(cln$Sample, names(ny)) 

betas <- as.data.frame(data.table::fread("results/betas-ppi_adj-pitt.tsv"))

ny.df <- ny[ny$V1 %in% gsub("^C", "", betas$clst), ]

####################################################
#                       main                       #  
####################################################

ny.mat <- reformat_data(ny.df, cln)

# also check if we just do prescence/absence because this is how the scores will be computed
df.med <- data.frame(clst = names(ny.mat[, 2:(ncol(ny.mat)-1)]),
                     med = unlist(lapply(ny.mat[, 2:(ncol(ny.mat)-1)], median)))

df.bin <- ny.mat %>%
  tidyr::pivot_longer(C142:C18821,
                      names_to = "clst") %>%
  select(name, clst, value) %>%
  inner_join(df.med, by = "clst") %>%
  mutate(seq.bin = ifelse(value >= med, 1, 0)) %>%
  select(name, clst, seq.bin) %>%
  tidyr::pivot_wider(names_from = clst,
                     values_from = seq.bin) %>%
  left_join(cln[, c("Sample", "Response")],
            by = c("name" = "Sample"))


# make the actual burden score now
# test all, risk, prt, & no prt (4 total)

# with betas
cd.mrs <- df.bin %>% 
  tidyr::pivot_longer(C142:C18821,
                      names_to = "clst") %>%
  inner_join(betas, by = "clst") %>%
  mutate(tot = value * beta) %>%
  select(name, Response, tot) %>%
  group_by(name, Response) %>%
  summarise(mrs = sum(tot)) %>%
  ungroup()


# try pruning & thresholding
cd.prt <- df.bin %>% 
  tidyr::pivot_longer(C142:C18821,
                      names_to = "clst") %>%
  inner_join(betas %>% filter(pvalue < 0.05),
             by = "clst") %>%
  mutate(tot = value * beta) %>%
  select(name, Response, tot) %>%
  group_by(name, Response) %>%
  summarise(mrs = sum(tot)) %>%
  ungroup()

risk.mrs <- df.bin %>% 
  tidyr::pivot_longer(C142:C18821,
                      names_to = "clst") %>%
  inner_join(betas %>% filter(beta < 0), by = "clst") %>%
  mutate(tot = value * beta) %>%
  select(name, Response, tot) %>%
  group_by(name, Response) %>%
  summarise(mrs = sum(tot)) %>%
  ungroup()


# try pruning & thresholding
risk.prt <- df.bin %>% 
  tidyr::pivot_longer(C142:C18821,
                      names_to = "clst") %>%
  inner_join(betas %>% filter(pvalue < 0.05 & beta < 0),
             by = "clst") %>%
  mutate(tot = value * beta) %>%
  select(name, Response, tot) %>%
  group_by(name, Response) %>%
  summarise(mrs = sum(tot)) %>%
  ungroup()


cd.mrs$pruning.thresholding <- "no"
cd.prt$pruning.thresholding <- "0.05"
risk.mrs$pruning.thresholding <- "no"
risk.prt$pruning.thresholding <- "0.05"
cd.mrs$type <- "combined"
cd.prt$type <- "combined"
risk.mrs$type <- "risk"
risk.prt$type <- "risk"


all.mrs <- rbind(cd.mrs, cd.prt, risk.mrs, risk.prt)

# get p value & auc for differences
pROC::auc(cd.mrs$Response, cd.mrs$mrs)
wilcox.test(cd.mrs$mrs[cd.mrs$Response == 1], 
            cd.mrs$mrs[cd.mrs$Response == 0],
            alternative = "greater")
pROC::auc(cd.prt$Response, cd.prt$mrs)
wilcox.test(cd.prt$mrs[cd.prt$Response == 1], 
            cd.prt$mrs[cd.prt$Response == 0],
            alternative = "greater")
pROC::auc(risk.mrs$Response, risk.mrs$mrs)
wilcox.test(risk.mrs$mrs[risk.mrs$Response == 1], 
            risk.mrs$mrs[risk.mrs$Response == 0],
            alternative = "greater")
pROC::auc(risk.prt$Response, risk.prt$mrs)
wilcox.test(risk.prt$mrs[risk.prt$Response == 1], 
            risk.prt$mrs[risk.prt$Response == 0],
            alternative = "greater")

# so we'll use the prt combined method

write.table(cd.prt, row.names = FALSE, quote = FALSE, sep = "\t",
            file = "results/mrs-ny.tsv")


