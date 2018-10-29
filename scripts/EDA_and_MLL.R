library(ggplot2)
library(reshape2)
census_df <- read.csv("Desktop/Penn/Jensen_Lab/phillybg.frame.2016.csv")
census_df[0,]
census_df$total_violent <- rowSums(data.frame(census_df[27:37]))
census_df$total_nonviolent <- rowSums(data.frame(census_df[38:48]))
                                  

## Make total histogram
ggplot(census_df, aes(x=total)) +
  labs(title="Total Population Counts in Block Groups",x="Total Population Counts", y="Frequency") +
  geom_histogram(color="black", fill="lightblue")

## Make pop histogram

pop_hist_df <- data.frame(counts=c(census_df$white, census_df$black, census_df$asian, census_df$hispanic, census_df$other),
                      pop=c(rep('white',length(census_df$white)), rep('black',length(census_df$black)), rep('asian',length(census_df$asian)), rep('hispanic',length(census_df$hispanic)), rep('other',length(census_df$other))))

ggplot(pop_hist_df, aes(x=counts, fill=pop,color=pop)) +
  labs(title="Population Counts in Block Groups",x="Counts Population Counts", y="Frequency") +
  geom_histogram(position="identity", alpha=0.1)

## Overall Pop Histogram
pop_prop_df <- data.frame(sums=c(sum(census_df$white), sum(census_df$black),sum(census_df$asian),sum(census_df$hispanic),sum(census_df$other)), pop=c('white', 'black', 'asian', 'hispanic', 'other'))
ggplot(pop_prop_df, aes()) +
  labs(title="Population Counts in Block Groups",x="Counts Population Counts", y="Frequency") +
  geom_bar(position="identity", alpha=0.1)

## Violent Crime v Nonviolent crime
violent_crime <- colSums(census_df[27:37])
nonviolent_crime <- colSums(census_df[38:48])
crime_df <- data.frame(violent_crime, nonviolent_crime, row.names = c(2006:2016))
crime_df

crime_df_v2_sums <- data.frame(group=rep(c("Violent", "Non-Violent"), each=11),
                          year=c(2006:2016,2006:2016),
                          count=c(violent_crime,nonviolent_crime))


ggplot(crime_df_v2_sums, aes(x=year,y=count,group=group)) +
  labs(title="Violent and Nonviolent crime in Philadelphia (2006-2016)",y="Total Crimes", x="Year") +
  geom_line(aes(color=group)) +
  geom_point()

viol_crime_df_ind <- melt(data.frame(census_df[27:37]))
viol_crime_df_ind$rowID <- 1:1336
viol_crime_df_ind$year <- rep(2006:2016, each=1336)

ggplot(viol_crime_df_ind, aes(x=year, y=value, group=rowID)) +
  geom_point(aes(color = rowID)) +
  geom_smooth(method=lm)

nonviol_crime_df_ind <- melt(data.frame(census_df[38:48]))
nonviol_crime_df_ind$rowID <- 1:1336
nonviol_crime_df_ind$year <- rep(2006:2016, each=1336)

ggplot(nonviol_crime_df_ind, aes(x=year, y=value, group=rowID)) +
  geom_line(aes(color = rowID)) #+
  #geom_smooth(method=lm)

##Crime Histograms
crime_per_block_df <- data.frame(violent=rowSums(data.frame(census_df[27:37])), nonviolent=rowSums(data.frame(census_df[38:48])))
crime_per_block_df$diff <- crime_per_block_df$nonviolent - crime_per_block_df$violent

ggplot(melt(crime_per_block_df), aes(x=value, fill=variable, color = variable)) +
  labs(title="Violent and Nonviolent Crime Histogram",x="# of Crimes", y="Frequency") +
  geom_histogram(position="identity",alpha = .1, binwidth = 100)

# compare violent with nonviolent crime
cor(crime_per_block_df$violent, crime_per_block_df$nonviolent)
violent_v_non_linear_model <- lm(data=crime_per_block_df, violent ~ nonviolent)
summary(violent_v_non_linear_model)
plot(violent_v_non_linear_model)
ggplot(crime_per_block_df, aes(nonviolent, violent)) +
  geom_point() + 
  geom_smooth(method = 'lm') + 
  labs(title="Violent vs Nonviolent crime")

ggplot(data.frame(diff=crime_per_block_df$diff), aes(x=diff)) +
  labs(title="Nonviolent minus Violent Crime Histogram",x="Crime Difference", y="Frequency") +
  geom_histogram(color="black", fill="lightblue", binwidth = 50)
ggplot(melt(crime_per_block_df), aes(x=variable, y=value)) +
  geom_boxplot()
  

# Multiple Linear Regression
#violent
big_lm_model <- lm(data = census_df, total_violent ~ area + poverty.1.verypoor + poverty.2.poor + poverty.3.middle + poverty.4.uppermiddle + segregationmetric + povertymetric + vacantprop + comresprop + whiteprop + blackprop + asianprop + otherprop + hispanicprop)

big_lm_nonviolent_model <- lm(data = census_df, total_nonviolent ~ area + poverty.1.verypoor + poverty.2.poor + poverty.3.middle + poverty.4.uppermiddle + segregationmetric + povertymetric + vacantprop + comresprop + whiteprop + blackprop + asianprop + otherprop + hispanicprop)

## scatter matrix
scatter_df = data.frame(total_violent=census_df$total_violent, area=census_df$area, verypoor=census_df$poverty.1.verypoor,poor=census_df$poverty.2.poor,middle_pov=census_df$poverty.3.middle, uppermiddle_pov=census_df$poverty.4.uppermiddle ,segregationmetric=census_df$segregationmetric ,povertymetric=census_df$povertymetric ,vacantprop=census_df$vacantprop ,comresprop=census_df$comresprop ,whiteprop=census_df$whiteprop ,blackprop=census_df$blackprop  ,asianprop=census_df$asianprop ,otherprop=census_df$otherprop ,hispanicprop=census_df$hispanicprop)
pairs(scatter_df[,], pch = 1)
scatter_df[scatter_df=='NA'] <- NA
scatter_df_none_missing <- scatter_df[rowSums(is.na(scatter_df))==0,]
subset_df <- data.frame(total_violent=log(census_df$total_violent), area=census_df$area, vacantprop=census_df$vacantprop ,comresprop=census_df$comresprop, segregationmetric=census_df$segregationmetric ,povertymetric=census_df$povertymetric)
#subset_df <- data.frame(total_nonviolent=census_df$total_nonviolent, area=census_df$area, vacantprop=census_df$vacantprop ,comresprop=census_df$comresprop, segregationmetric=census_df$segregationmetric ,povertymetric=census_df$povertymetric)
subset_df_none_missing <- subset_df[rowSums(is.na(subset_df))==0,]

# Correlation panel
panel.cor <- function(x, y){
  usr <- par("usr"); on.exit(par(usr))
  par(usr = c(0, 1, 0, 1))
  r <- round(cor(x, y), digits=2)
  txt <- r
  cex.cor <-0.8/strwidth(txt)
  text(0.5, 0.5, txt)
}
# Customize upper panel
upper.panel<-function(x, y){
  points(x,y, pch = 20, cex = 1)
}
# Create the plots
pairs(subset_df_none_missing[,], 
      lower.panel = panel.cor,
      upper.panel = upper.panel)

