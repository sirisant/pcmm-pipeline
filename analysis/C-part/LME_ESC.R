library(lmerTest)

df <- read.csv("pcmm_categorical_states.csv")

df$Subject     <- as.factor(df$Subject)
df$Task_Name   <- as.factor(df$Task_Name)
df$Brain_State <- as.factor(df$Brain_State)

# R will automatically select State_1 as the reference intercept 
model <- lmer(ESC ~ Brain_State + (1 | Subject:Task_Name), data = df)

summary(model)

rmse_val <- sqrt(mean(residuals(model)^2))
print(rmse_val)