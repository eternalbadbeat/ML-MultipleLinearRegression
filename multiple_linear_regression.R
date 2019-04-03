#Data preprocessing 

#Imoporting the dataset 
dataset = read.csv('50_Startups.csv')
#dataset = dataset[, 2:3]

#splitting the dataset into the training set and test set
#библиотека для сплита - CaTools
#install.packages('caTools')
#устанавливаем рандомное число для сплита
set.seed(123)

#Encoding categorail data 
dataset$State = factor(dataset$State,
                         levels = c('New York', 'California', 'Florida'),
                         labels = c(1,2,3))

#в питоне нам надо было указывать и матрицу Х и зависимый ветор У, здесь 
#нам надо указать только столбец зависимой переменной 
#в splitrartio мы просто указываем, процент набора данных для тренировки
split = sample.split(dataset$Profit, SplitRatio = 0.8)
#после выполнения строчки сверху, мы получим значение split, которое будет состоять из тру и фолс 
#тру - попадание в трейнинг сет, фолс - в тест сет 

#теперь нам надо отдельно создать тест сет и трейнинг сет
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

#fit multiple linear regression to the training_set
#Profit ~ R.D.Spend + Administration + Marketing.Spend + State равно Profit ~ .
#точка означает, что мы берем все независимые переменные
regressor = lm(formula = Profit ~ ., 
               data = training_set )

#predict the test_set results 

y_pred = predict(regressor, newdata = test_set)

#optimal model with backward elimination

regressor = lm(formula = Profit ~ R.D.Spend + Administration + Marketing.Spend + State, 
               data = dataset )

summary(regressor)

regressor = lm(formula = Profit ~ R.D.Spend + Administration + Marketing.Spend, 
               data = dataset )

summary(regressor)

regressor = lm(formula = Profit ~ R.D.Spend + Marketing.Spend, 
               data = dataset )

summary(regressor)

regressor = lm(formula = Profit ~ R.D.Spend, 
               data = dataset )

summary(regressor)






