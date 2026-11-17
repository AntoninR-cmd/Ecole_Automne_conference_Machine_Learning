# dependencies
library(tidyverse)

# load all domestic functions
source("R/vgg_and_eft_this.R")

# Graphics
# days <- 180
# library(lubridate)
library(fs)
# grab all results from .rda files
list.files("rda/results", rec=TRUE, full=TRUE) %>% 
  fs::file_info() %>% 
  # select(path, modification_time) %>% 
  # mutate(days_old=(Sys.time() - modification_time) %>% as.numeric(units="days")) %>% 
  # filter(days_old < days) %>%
  pull(path) %>% 
  map(rda2list) %>% 
  map_dfr(~.x[[1]]) %>% mutate(data=str_remove(data, "_db")) -> res_df
save(res_df, file="rda/res.rda")

load("rda/res.rda")
# graph this
res_df %>% 
  mutate(view=str_replace(view, "VV", "VD")) %>% 
  select(seed, data, view, N, eft=eft_external, cnn=deep_external_acc) %>% 
  pivot_longer(cols=c(eft, cnn), names_to = "method", values_to = "acc") -> res_df_ready 

res_df_ready %>% 
  mutate(data_view=paste0(data, "_", view),
         data_view_seed_meth=paste0(data, "_", view, "_", seed, "_", method)) %>% 
  # arrange(data_view_seed, N) %>% 
  ggplot() +
  aes(x=N, y=acc*100, group=data_view_seed_meth, col=method) +
  geom_point(size=0.5) +
  geom_line(size=0.2) +
  # geom_line(size=0.2, alpha=0.5) +
  # geom_smooth(method="loess", formula="y~x", span=0.5,
  #             se=F, size=0.5, alpha=0.2) +
  # # geom_smooth(method="lm", formula="y~log(x)",
  #             se=F, size=0.25, alpha=0.2) +
  # geom_boxplot() +
  facet_grid(data~view, scales = "free_y") +
  # facet_wrap(~data_view)+
  # scale_y_continuous(minor_breaks = 0.5) +
  scale_x_continuous(breaks=seq(0, 700, 100)) +
  scale_color_viridis_d(begin = 0.3, end=0.7, option = "C") +
  xlab("sample size in training set") + 
  ylab("accuracy (%)") +
  theme_minimal() +
  theme(panel.grid.minor=element_blank(),
        panel.background = element_rect(fill=NULL, colour="grey50"),
        axis.text.x = element_text(angle=60, hjust=1),
        strip.text = element_text(size = 10, face="bold"))
ggsave("gr/acc.pdf", width=6, height=6)


res_df %>% 
  mutate(view=str_replace(view, "VV", "VD")) %>% 
  select(seed, data, view, N, eft=eft_external, cnn=deep_external_acc) %>% 
  group_by(seed, data, view, N) %>% 
  mutate(diff=100*(cnn-eft)) %>% 
  select(-cnn, -eft) %>% 
  ungroup() -> z

z %>% 
  group_by(data, view, N) %>% 
  summarise(mean_diff=mean(diff)) %>% 
  ungroup() %>% 
  mutate(mean_diff=round(mean_diff, 1)) %>% 
  pivot_wider(names_from=N, values_from = mean_diff) %>% 
  write.table(file="paper/Table2_pre.csv", row.names = FALSE, sep=";")

z %>%
  group_by(data, view, N) %>% 
  summarise(better=sum(diff>0)) %>% ungroup() %>% 
  pivot_wider(names_from=N, values_from = better)

res_df %>% 
  mutate(view=str_replace(view, "VV", "VD")) %>% 
  select(seed, data, view, N, eft=eft_external, cnn=deep_external_acc) %>% 
  group_by(data, view) %>% 
  filter(N==max(N)) %>% 
  group_by(data, view) %>% 
  summarise(eft=mean(eft), cnn=mean(cnn))

# table summary
z %>%
  group_by(data, view, N) %>% 
  summarise(better=sum(diff>0), n=n()) %>% 
  group_by(data, view) %>% 
  filter(N==max(N)) %>% ungroup()




# table summary
z %>% group_by(data, view, N) %>% 
  summarise(better=sum(diff>0), n=n()) %>% 
  group_by(data, view) %>% 
  filter(N==max(N)) %>% ungroup()


z %>% ggplot() + 
  aes(N, diff, group=seed) + 
  geom_hline(yintercept=0, col="red", size=0.5, linetype="dashed") +
  geom_smooth(aes(group=NULL, col=NULL), method="loess", formula="y~x",
              span=3, col="blue", size=0.5, se=FALSE) +
  geom_line(size=0.25, alpha=0.5) +
  geom_point(size=0.5) +
  
  xlab("sample size in training set") + 
  ylab("(eft - cnn) accuracy (%)") +
  facet_grid(data~view, scales = "free_y") + 
  scale_x_continuous(breaks=c(50, 100, 150, 200, 300, 400, 500, 600, 700)) +
  scale_y_continuous(breaks=seq(-20, 20, 10), limits = c(-20, 20)) +
  scale_color_viridis_d(option="plasma", end=0.7) +
  theme_minimal() +
  theme(legend.position = "none",
        panel.background = element_rect(fill=NULL, colour="grey20", size=0.5),
        panel.grid.minor=element_blank(),
        panel.grid.major = element_line(colour="grey80", size=0.05),
        axis.text.x = element_text(angle=60, hjust=1),
        strip.text = element_text(size = 10, face="bold"))
ggsave("gr/diff.pdf", height=9, width=4)

res_df %>% 
  mutate(view=str_replace(view, "VV", "VD")) %>% 
  select(seed, data, view, N, eft=eft_external, cnn=deep_external_acc) %>% 
  group_by(seed, data, view) %>% 
  arrange(desc(N)) %>% slice(1) %>% 
  ungroup() %>% 
  pivot_longer(cols=c(eft, cnn), names_to = "method", values_to = "acc") %>% 
  arrange(seed, data, view, method) %>% 
  group_by(seed, data, view) %>% 
  mutate(diff=100*(acc[1]-acc[2])) %>% slice(1) %>% 
  ungroup() %>% 
  select(seed, data, view, diff) %>% 
  unite("dataset", c(data, view), sep="_") %>% 
  ggplot() + 
  aes(dataset, diff) + 
  geom_col(position="identity") +
  geom_point() +
  xlab("sample size in training set") + 
  ylab("accuracy (%)") +
  theme_minimal() +
  theme(panel.grid.minor=element_blank(),
        panel.background = element_rect(fill=NULL, colour="grey50"),
        axis.text.x = element_text(angle=60, hjust=1),
        strip.text = element_text(size = 10, face="bold"))


# 1out vs eft_external
res_df %>%
  select(data:eft_external, -N) %>% 
  mutate(d=100*(eft_1out - eft_external)) %>% 
  ggplot() + aes(d) + geom_histogram()
# 
# # 75
# vgg16_and_eft6_this(hordeum_db,  data_col="VL_deep",
#                     N_model=200, epochs = 100, seed=2329)
# 

# TABLE 1: material used
hordeum_db %>% select(status) %>% count(status)

list(hordeum=hordeum_db,
     olea=olea_db, 
     phoenix=phoenix_db,
     vitis=vitis_db) %>%
  imap_dfr(~.x %>% select(status) %>% count(status) %>% mutate(taxa=.y)) %>% 
  select(taxa, status, n) %>% 
  write.table(file="paper/Table1_pre.csv", row.names = FALSE, sep=";")

