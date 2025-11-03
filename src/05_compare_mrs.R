####################################################
#                      readme                      #  
####################################################

# validate risk score for clinical response

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


cln <- as.data.frame(data.table::fread('supp_dataset_1.csv', header = TRUE))
cln$Response <- ifelse(cln$Study_Clin_Response == "Responder", 0, 1)
cln$BMI <- as.numeric(cln$BMI)

all.dfs <- list()
all.dfs$Pittsburgh = as.data.frame(data.table::fread("supp_dataset_4-$Pitt.csv"))
all.dfs$NewYork = as.data.frame(data.table::fread("supp_dataset_4-$NY.csv"))
all.dfs$Dallas = as.data.frame(data.table::fread("supp_dataset_4-$Dallas.csv"))
all.dfs$Houston = as.data.frame(data.table::fread("supp_dataset_4-$Houston.csv"))

betas <- as.data.frame(data.table::fread("results/betas-ppi_adj-pitt.tsv"))

dfs <- lapply(all.dfs, function(x) {y = x[x$V1 %in% gsub("^C", "", betas$clst), ]; return(y)})

####################################################
#                       main                       #  
####################################################

mats <- lapply(dfs, function(x) reformat_data(x, cln))

# do prescence/absence because this is how the scores will be computed
df.meds <- lapply(mats, function(x) 
  {y = data.frame(clst = names(x[, 2:(ncol(x)-1)]),
                  med = unlist(lapply(x[, 2:(ncol(x)-1)], median)))
  return(y)})

df.bins <- Map(function(x, y){
  z <- x %>%
    tidyr::pivot_longer(C260:C21467,
                        names_to = "clst") %>%
    select(name, clst, value) %>%
    inner_join(y, by = "clst") %>%
    mutate(seq.bin = ifelse(value >= med, 1, 0)) %>%
    select(name, clst, seq.bin) %>%
    tidyr::pivot_wider(names_from = clst,
                       values_from = seq.bin) %>%
    left_join(cln[, c("Sample", "Response")],
              by = c("name" = "Sample"))
  return(z)
}, mats, df.meds)
 
# make the actual burden score now
# test all, risk, prt, & no prt (4 total)

# with betas
cd.mrs <- lapply(df.bins, function(x){
  y = x %>% 
    tidyr::pivot_longer(C260:C21467,
                        names_to = "clst") %>%
    inner_join(betas, by = "clst") %>%
    mutate(tot = value * beta) %>%
    select(name, Response, tot) %>%
    group_by(name, Response) %>%
    summarise(mrs = sum(tot)) %>%
    ungroup()
  return(y)
})


# try pruning & thresholding
cd.prt <- lapply(df.bins, function(x){
  y <- x %>%
    tidyr::pivot_longer(C260:C21467,
                        names_to = "clst") %>%
    inner_join(betas %>% filter(pvalue < 0.05),
               by = "clst") %>%
    mutate(tot = value * beta) %>%
    select(name, Response, tot) %>%
    group_by(name, Response) %>%
    summarise(mrs = sum(tot)) %>%
    ungroup()
  return(y)
})


# just risk
risk.mrs <- lapply(df.bins, function(x){
  y <- x %>%
    tidyr::pivot_longer(C260:C21467,
                        names_to = "clst") %>%
    inner_join(betas %>% filter(beta > 0), by = "clst") %>%
    mutate(tot = value * beta) %>%
    select(name, Response, tot) %>%
    group_by(name, Response) %>%
    summarise(mrs = sum(tot)) %>%
    ungroup()
  return(y)
})


# try pruning & thresholding
risk.prt <- lapply(df.bins, function(x){
  y <- x %>%
    tidyr::pivot_longer(C260:C21467,
                        names_to = "clst") %>%
    inner_join(betas %>% filter(pvalue < 0.05 & beta > 0),
               by = "clst") %>%
    mutate(tot = value * beta) %>%
    select(name, Response, tot) %>%
    group_by(name, Response) %>%
    summarise(mrs = sum(tot)) %>%
    ungroup()
  return(y)
})

# just response
resp.mrs <- lapply(df.bins, function(x){
  y <- x %>%
  tidyr::pivot_longer(C260:C21467,
                      names_to = "clst") %>%
    inner_join(betas %>% filter(beta < 0), by = "clst") %>%
    mutate(tot = value * beta) %>%
    select(name, Response, tot) %>%
    group_by(name, Response) %>%
    summarise(mrs = sum(tot)) %>%
    ungroup()
  return(y)
})


# try pruning & thresholding
resp.prt <- lapply(df.bins, function(x){
  y <- x %>%
    tidyr::pivot_longer(C260:C21467,
                        names_to = "clst") %>%
    inner_join(betas %>% filter(pvalue < 0.05 & beta < 0),
               by = "clst") %>%
    mutate(tot = value * beta) %>%
    select(name, Response, tot) %>%
    group_by(name, Response) %>%
    summarise(mrs = sum(tot)) %>%
    ungroup()
  return(y)
})

all.mrs <- list()
all.mrs$MRS.Full <- bind_rows(cd.mrs, .id= "id")
all.mrs$MRS.Full$type <- "combined"
all.mrs$MRS.PRT <- bind_rows(cd.prt, .id= "id")
all.mrs$MRS.PRT$type <- "combined"
all.mrs$Risk.Full <- bind_rows(risk.mrs, .id= "id")
all.mrs$Risk.Full$type <- "risk"
all.mrs$Risk.PRT <- bind_rows(risk.prt, .id= "id")
all.mrs$Risk.PRT$type <- "risk"
all.mrs$Response.Full <- bind_rows(resp.mrs, .id= "id")
all.mrs$Response.Full$type <- "resp"
all.mrs$Response.PRT <- bind_rows(resp.prt, .id= "id")
all.mrs$Response.PRT$type <- "resp"

mrs.full <- bind_rows(all.mrs, .id = "method")

lapply(cd.mrs, function(x) pROC::auc(x[["Response"]], x[["mrs"]]))
lapply(cd.prt, function(x) pROC::auc(x[["Response"]], x[["mrs"]]))
lapply(risk.mrs, function(x) pROC::auc(x[["Response"]], x[["mrs"]]))
lapply(risk.prt, function(x) pROC::auc(x[["Response"]], x[["mrs"]]))
lapply(resp.mrs, function(x) pROC::auc(x[["Response"]], x[["mrs"]]))
lapply(resp.prt, function(x) pROC::auc(x[["Response"]], x[["mrs"]]))


mrs.full$id <- factor(mrs.full$id, levels = c("Pittsburgh", "NewYork", "Dallas", "Houston"))

aurocs <- list()
ps <- list()
# get p value & auc for differences
for(i in unique(mrs.full$method)){
  print(i)
  aurocs[[i]] <- lapply(names(mats), function(x) (print(pROC::auc(mrs.full$Response[mrs.full$id == x & mrs.full$method == i], 
                                                                  mrs.full$mrs[mrs.full$id == x & mrs.full$method == i]))))
  ps[[i]] <- lapply(names(mats), function(x) (wilcox.test(mrs.full$mrs[mrs.full$id == x & mrs.full$method == i & mrs.full$Response == 1],
                                                          mrs.full$mrs[mrs.full$id == x & mrs.full$method == i & mrs.full$Response == 0],
                                                          alternative = "greater")$p.value))
}



