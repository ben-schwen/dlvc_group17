library(data.table)
library(ggplot2)

setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

data = fread('result.csv')
colnames(data) <- c("AdaBins", "MiDaS small", "MiDaS hybrid", "MiDaS large")
data[, id := .I]

long = melt(data, id.vars = c("id"), measure.vars = c("AdaBins", "MiDaS small", "MiDaS hybrid", "MiDaS large"))
colnames(long) <- c("picture_id", "Model","RMSE log")

line <- ggplot(long, aes(x = picture_id, y = `RMSE log`, colour = Model)) +
  geom_line() + 
  theme_update(text = element_text(size=40)) +
  theme_minimal()
line

box <- ggplot(long, aes(y = `RMSE log`, colour = Model)) +
  geom_boxplot() + 
  theme_update(text = element_text(size=40)) +
  theme_minimal() 

png('model_line.png')
plot(line)
dev.off()

png('model_box.png')
plot(box)
dev.off()

models <- c("AdaBins", "MiDaS small", "MiDaS hybrid", "MiDaS large")
data[, lapply(.SD, mean), .SDcols = models]
data[, lapply(.SD, sd), .SDcols = models]
