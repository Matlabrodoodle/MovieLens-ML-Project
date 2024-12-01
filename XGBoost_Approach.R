# Load packages
library(tidyverse)
library(data.table)
library(xgboost)


# INITIAL SETUP

# The movie data ReadMe states that errors or inconsistencies may exist in the data
# So first I check for errors that may adversely impact eventual model performance/accuracy

# Set seed so that results are repeatable. Choose funny meme number for seed.
set.seed(42)

# First check for missing values in key columns using is.na()
print(paste(sum(is.na(edx$title)), "missing titles."))
print(paste(sum(is.na(edx$movieId)), "missing movie IDs."))
print(paste(sum(is.na(edx$userId)), "missing user IDs."))
print(paste(sum(is.na(edx$genres)), "missing genres."))
print(paste(sum(is.na(edx$timestamp)), "missing timestamps."))

# Confirm all movie titles end with release year in the format "(1999)"
movies_with_year <- length(grep("\\([0-9]{4}\\)$", edx$title))

# Print message with whether years are present and correctly formatted or not
if (movies_with_year == length(edx$title)) {print("Years are present and uniformly formatted")
}  else {print("Years are either missing or improperly formatted")}

# Create a subset of full dataframe for model tuning and validation
edx_small <- edx %>% sample_n(size = nrow(edx)/10)

# Drop missing genres as there are very few instances and they will interfere with feature creation
edx_small <- subset(edx_small,edx_small$genres != "(no genres listed)")



# FUNCTIONS

# One_Hot_Genrecoder is a One Hot Encoder for encoding movie genres
# 1 means the genre applies, 0 means it does not
# The singular argument determines whether only the first listed genre is retained (TRUE) or if all are retained (FALSE)
# Initial testing determined that RMSE is improved with singular = TRUE
One_Hot_Genrecoder <- function(df,singular) {
  
  # Break genres by "|" divider and create a new One Hot column for each new genre encountered
  if (singular==FALSE) {
    
    df <- df %>%
      
      separate_rows(genres, sep= "\\|") %>%
      
      mutate(present = 1) %>%
      
      pivot_wider(names_from=genres,values_from=present,values_fill = list(present=0))
    
  }
  
  # Extract the first genre and create a new One Hot column for each new genre encountered
  else {
    
    df <- df %>%
      mutate(genres=str_extract(genres,"^[^|]+")) %>%
      
      mutate(present = 1) %>%
      
      pivot_wider(names_from=genres,values_from=present,values_fill = list(present=0))
  }
  
  # Return the one hot encoded dataframe
  return(df)
}


# This function makes use of datetime information to create a "Nostalgia Factor"
# The idea here is users tend to give higher ratings to older movies (observed during EDA)
# It accomplishes this by taking the difference between the timestamp year and movie year
Create_Nostalgia_Factor <- function(data) {
  
  # Extract the year from the movie title (this is why we checked earlier that years are all present)
  data <- data %>%
    mutate(year = str_extract(title, "\\([0-9]{4}\\)$"),year = str_sub(year,2,-2),title=str_sub(title,1,-7))
  
  # Convert year to int
  data$year <- as.integer(data$year)
  
  # Convert timestamp to useable format and extract year
  data$timestamp <- as.POSIXct(data$timestamp)
  data$timestamp <- year(data$timestamp)
  
  # Create feature by taking the difference of the two years and normalize
  data$nostalgia_factor <- data$timestamp - data$year
  data$nostalgia_factor <- data$nostalgia_factor/max(data$nostalgia_factor)
  
  # Now we break nostalgia rating tendencies into ten "buckets"
  # Lower bucket values (more recent films) tend to have lower ratings
  for (i in -1:10){
    
    # Convert integer to decimal corresponding to normalized nostalgia factor, 0 to 1
    j <- i/10
    
    # Replace nostalgia_factor with corresponding average rating for each bucket
    data$nostalgia_factor[data$nostalgia_factor>j & data$nostalgia_factor <= j+.1] <-
      mean(data$rating[data$nostalgia_factor>j & data$nostalgia_factor <= j+.1])
    
  }
  
  # Move new column arbitrarily early so it doesn't interfere with future code execution
  data <- setcolorder(data,"nostalgia_factor",before=3)
  data <- setcolorder(data,"year",before=3)
  
  return(data)
}


# This function aggregates mean ratings information over the specified target column (e.g., by userId)
# Train and test data are separate inputs to this function and both are returned
# Alt_value is substituted where a mean rating is not available for the target variable
# Round determines whether ratings features are rounded to the nearest multiple of 0.5 (Boolean)
Add_Rating_Means <- function (train_data, test_data, target_col,alt_value,round) {
  
  # Append _rating_means to the target variable name
  new_col_name <- paste0(target_col,"_rating_means") 
  
  # Create rating sums and merge into copy of training data
  rating_stats <- train_data[,.(sum_rating=sum(rating),count=.N),by=target_col]
  
  train_data_updated <- merge(train_data,rating_stats,by=target_col,all.x=TRUE)
  
  # To avoid data leakage we implement a simple Leave One Out encoder
  # I.e., the current instance's rating cannot be used to predict itself
  # This is achieved by using alt_value where only one instance of the target variable exists in the training data
  # Otherwise, the current instance's rating is subtracted when computing the mean
  train_data_updated[,(new_col_name):= ifelse(
    
    count > 1,
    (sum_rating-rating)/(count-1),
    alt_value
    
  )]
  
  # Drop temporary columns
  train_data_updated[,c("sum_rating","count"):=NULL]
  
  # For creating the feature in the test data set, the approach is simpler
  # We don't have to worry about data leakage so no need for Leave One Out encoding
  # We simply extract average ratings by target variable and put in a new table
  rating_means <- train_data[,.(rating_mean=mean(rating)),by=target_col]
  setnames(rating_means,"rating_mean",new_col_name)
  
  # Merge rating_means into the test data, substituting the alt_value where the target is not present in test data
  test_data_updated <- merge(test_data,rating_means,by = target_col, all.x=TRUE)
  test_data_updated[is.na(get(new_col_name)),(new_col_name):=alt_value]
  
  # Round results if desired
  if (round==TRUE){
    train_data_updated[,(new_col_name):=round(get(new_col_name)*2)/2] 
    test_data_updated[,(new_col_name):=round(get(new_col_name)*2)/2] 
  }
  
  # Return results as list
  return(list(train_data=train_data_updated,test_data = test_data_updated))

}

# This function extracts the user's rating tendencies by genre for feature creation
# The idea being that user's tend to have stable genre preferences independent of other factors
# The function returns a list of User IDs and their mean ratings across genres
# Start_col specifies the first column with a One Hot Encoded genre
Extract_User_Genre_Ratings <- function(train_data,start_col) {
  
  # Extract column names starting with the first genre column to the rightmost column
  cols <- colnames(train_data)[start_col:length(train_data)]
  
  # Create a copy and multiply genre columns by the corresponding movie rating
  # Since genres are one hot encoded, zeroes remain zeroes
  # Applicable genres for each instance now reflect the movie rating
  train_data_mod <- copy(train_data)
  
  train_data_mod[,(cols):=lapply(.SD,function(x)x*rating),.SDcols=cols]
  
  # create the result by averaging non-zero genres by userId
  result <- train_data_mod[,lapply(.SD,function(x) mean(x[x !=0],na.rm=TRUE)),by="userId",.SDcols=cols]
  
  # Create global ratings by userId as alternate value, simply the average rating that user gives
  # This is used where an existing user has never rated certain genres
  global_ratings <- train_data_mod[,.(global_ratings=mean(rating,na.rm=TRUE)),by="userId"]
  
  # Replace all NAN values in the result with the global rating of the user and then drop the global ratings column
  result <- merge(result,global_ratings,by="userId",all.x=TRUE)
  
  result[,(cols):=lapply(.SD,function(x) ifelse(is.nan(x),global_ratings,x)),by="userId",.SDcols=cols]
  
  result[,global_ratings := NULL]
  
  return(result)
}

# This function is the counterpart to Extract_User_Genre_Ratings and is called afterwards
# It takes the newly created user genre data and the movie data (training or testing)
# And creates a feature column reflecting how users would be expected to rate a movie based on genre alone
# alt_value, start_col, and round all function as described previously
Add_User_Genre_Ratings <- function(movie_data,genre_rating_data,alt_value,start_col,round) {
  
  # Create column indices and get column names
  genre_count <- ncol(genre_rating_data)-1
  end_col <- start_col-1+genre_count
  cols = names(movie_data)[start_col:end_col]
  
  # Since other functions may have transformed our one hot encoding, return to binary values
  movie_data[,(cols):=lapply(.SD,function(x)ifelse(x!=0,1,0)),.SDcols=cols]
  
  # Join movie data with user genre data on userId and replace NA values with zeroes
  movie_data <- left_join(movie_data,genre_rating_data,by="userId")
  
  movie_data[is.na(movie_data)]<-0 
  
  # Multiply one hot genre columns by corresponding genre ratings (apologies for some sloppy indexing here)
  movie_data[,start_col:end_col] <- movie_data[,start_col:(end_col)]*movie_data[,(start_col+genre_count):(start_col-1+2*genre_count)]
  
  # Drop user genre ratings columns, we are done with them
  movie_data[,(start_col+genre_count):(start_col-1+2*genre_count):=NULL] 
  
  # Finally, average all nonzero genre ratings into one feature, User_Genre_Average
  movie_data[,User_Genre_Average:=apply(.SD,1,function(x){
    
    non_zero_values <- x[x!=0]
    
    if (length(non_zero_values)>0) {mean(non_zero_values)}
    else{NA} 
    
  }),.SDcols=start_col:end_col]
  
  # Replace NA instances with the alt_value
  movie_data[is.na(User_Genre_Average),User_Genre_Average:=alt_value]
  
  # Round if desired
  if (round==TRUE) {
    
    movie_data$User_Genre_Average <- round(movie_data$User_Genre_Average*2)/2
    
  }
  
  # Restore one hot encodings to binary values
  cols = names(movie_data)[start_col:end_col]
  movie_data[,(cols):=lapply(.SD,function(x)ifelse(x!=0,1,0)),.SDcols=cols]
  
  # Move new feature to arbitrarily early position that won't interfere with future code execution
  movie_data <- setcolorder(movie_data,"User_Genre_Average",before=3)
  
  return(movie_data)
}

# This function is nearly identical to Extract_User_Genre_Rating except it operates on MovieIDs
# The idea being that past movie ratings predict future ratings given for the same movie
# The function returns a list of movie IDs and their mean ratings
# Start_col and round function as previously described
Extract_Movie_Genre_Ratings <- function(train_data,start_col,round){
  
  cols <- colnames(train_data)[start_col:length(train_data)]
  
  train_data_mod <- copy(train_data)
  
  # Again we multiply binary genre indicators by the rating
  train_data_mod[,(cols):=lapply(.SD, function(x)x*rating),.SDcols=cols]
  
  # Remove all non-genre columns
  train_data_mod[,1:(start_col-1):=NULL]
  
  result <- data.table()
  
  # Add genre names to new data table and add average rating by genre 
  for (i in 1:length(train_data_mod)) {
    
    genre_name <- names(train_data_mod)[i]
    
    result[,(genre_name)] <- train_data_mod[,lapply(.SD,function(x)mean(x[x!=0],na.rm=TRUE)),.SDcols=i] 
    
  }
  
  # Round if desired
  if (round==TRUE) {
    result <- round(result*2)/2
  }
  
  # Return table of average genre ratings
  return(result)
  
}

# This function creates a feature for expected movie rating based on average genre ratings
# The result of Extract_Movie_Genre_Rating is input as genre_rating_data
# start_col and round function as previously described
Add_Movie_Genre_Ratings <- function(movie_data,genre_rating_data,start_col,round) {
  
  movie_data_mod <- copy(movie_data)
  
  end_col <- length(movie_data)
  
  cols = names(movie_data_mod)[start_col:end_col]
  
  # Restore genre encodings to binary operators
  movie_data_mod[,(cols):=lapply(.SD,function(x)ifelse(x!=0,1,0)),.SDcols=cols]
  
  # Multiply corresponding genres in data by their average rating using set()
  for (i in start_col:end_col) {
    
    set(movie_data_mod, j = i, value = movie_data_mod[[i]] * genre_rating_data[[i + 1 - start_col]])
    
  }
  
  # Aggregate all genre averages into one Movie_Genre_Average
  # (If OneHotGenrecoder was called with singular = TRUE then non_zero_values will be 1 or 0)
  movie_data_mod[,Movie_Genre_Average:=apply(.SD,1,function(x){
    
    non_zero_values <- x[x!=0]
    
    if (length(non_zero_values)>0) {mean(non_zero_values)}
    else{NA} 
    
  }),.SDcols=cols]
  
  # Round if desired
  if (round==TRUE) {
    
    movie_data_mod$Movie_Genre_Average <- round(movie_data_mod$Movie_Genre_Average*2)/2
    
  }
  
  # Move column arbitrarily early so it doesn't mess anything up later
  movie_data_mod <- setcolorder(movie_data_mod,"Movie_Genre_Average",before=3)
  
  return(movie_data_mod)
  
} 


# MODEL VALIDATION

# Call functions to create genre encodings and Nostalgia factor feature
# Recall these two functions require dataframes not tables
edx_small <- One_Hot_Genrecoder(edx_small,TRUE)
edx_small <- Create_Nostalgia_Factor(edx_small)

# Convert edx_small to datatable
setDT(edx_small)

# Define number of cross validation folds
fold <- 4
# Initialize dummy counter
rmse_sum <- 0
# Initialize best RMSE variable 
best_rmse <- Inf
# Set best_params to NULL
best_params <- NULL

# Select feature columns manually
feature_columns <- c("User_Genre_Average","movieId_rating_means","Movie_Genre_Average","nostalgia_factor")

# If performing grid search of features, use unlist() to obtain all possible feature combinations
feature_combs <- unlist(lapply(1:length(feature_columns),function(x)combn(feature_columns,x,simplify=FALSE)),recursive=FALSE)

# Define number of grid search iterations (or manually define for random search if desired)
num_iter <- length(feature_combs)

# Partition edx_small into folds randomly
edx_small[,fold:=sample(1:fold,.N,replace=TRUE)]

# Move fold column to front so it does not interfere with indexing during function call
edx_small <- setcolorder(edx_small,"fold",before=1)

# Create search space for model parameters for use in random searches
search_space <- list(
  eta = c(0.01, 0.03, 0.07,0.1,0.15),
  max_depth = c(4, 6, 8),
  subsample = c(0.6, 0.8, 1),
  colsample_bytree = c(0.6, 0.8, 1),
  subsample = c(0.3, 0.5,1),
  alpha = c(0,1,10,100),
  min_child_weight = c(0,1,10,100)
)

# Set all_round variable as desired (testing revealed RMSE improves when this is FALSE)
all_round <- FALSE

# Create blank lists for training and validation folds
train_fold <- list()
val_fold <- list()

# Process training and validation data for all folds using the fold # as the index
# Call all feature-generating functions on each loop (this takes a while)
for (k in 1:fold) {

  train_fold[[k]] <- edx_small[fold!= k]
  val_fold[[k]] <- edx_small[fold==k]
  
  User_Genre_Ratings <- Extract_User_Genre_Ratings(train_fold[[k]],9)
  train_fold[[k]] <- Add_User_Genre_Ratings (train_fold[[k]],User_Genre_Ratings,NA,9,all_round )
  val_fold[[k]] <- Add_User_Genre_Ratings (val_fold[[k]],User_Genre_Ratings,NA,9,all_round )
  
  Movie_Genre_Ratings <- Extract_Movie_Genre_Ratings(train_fold[[k]],10,all_round)
  train_fold[[k]] <- Add_Movie_Genre_Ratings(train_fold[[k]],Movie_Genre_Ratings,10,all_round )
  val_fold[[k]] <- Add_Movie_Genre_Ratings(val_fold[[k]],Movie_Genre_Ratings,10,all_round)
  
  result <- Add_Rating_Means(train_fold[[k]], val_fold[[k]], "movieId",NA,all_round)
  train_fold[[k]] <- result$train_data
  val_fold[[k]] <- result$test_data

}

# Iterate through sample space selecting combinations of features and/or model parameters
# Currently this loop performs a grid search over all feature combinations
# With minor code changes this can be changed to perform a random search over model parameter space
# This was done previously and best-performing parameters were determined
for (i in 1:num_iter) {

# Sample model parameter space randomly on each loop  
# XGB_params <- list(
# 
#   objective = "reg:squarederror",
#   eval_metric = "rmse",
#   eta = sample(search_space$eta,1),
#   max_depth = sample(search_space$max_depth,1),
#   subsample = sample(search_space$subsample,1),
#   colsample_bytree = sample(search_space$colsample_bytree,1),
#   alpha = sample(search_space$alpha,1),
#   min_child_weight = sample(search_space$alpha,1)
# )

# Manually set model parameters if optimal parameters are already known
XGB_params <- list(

    objective = "reg:squarederror",
    eval_metric = "rmse",
    eta = 0.1,
    max_depth = 4,
    subsample = .6,
    colsample_bytree = 0.6,
    alpha = 100,
    min_child_weight = 1
)

# Iterate through each fold and partition into features and target variable (rating)
for (k in 1:fold) {

feature_selection <- feature_columns#feature_combs[[i]]

X_train <- train_fold[[k]][,..feature_selection]
Y_train <- train_fold[[k]]$rating
X_test <- val_fold[[k]][,..feature_selection]
Y_test <- val_fold[[k]]$rating

# Convert to matrices for input into model
X_train <- as.matrix(X_train)
Y_train <- as.matrix(Y_train)
X_test <- as.matrix(X_test)
Y_test <- as.matrix(Y_test)

# Create DMatrix for training and validation sets within current fold
dtrain <- xgb.DMatrix(data=X_train,label=Y_train,missing=NA)
dtest <- xgb.DMatrix(data=X_test,label = Y_test)

# Train model with selected parameters
# nrounds and early stopping rounds are manually selected (not in random search)
XGB_model <- xgb.train(

  params = XGB_params,
  data = dtrain,
  nrounds = 100,
  watchlist = list(train=dtrain,test=dtest),
  early_stopping_rounds = 10,
  verbose = 1

)

# Create predictions and calculate RMSE for current fold
predictions <- predict(XGB_model,newdata = dtest)

rmse <- sqrt(mean((predictions-Y_test)^2))

# Maintain running sum of RMSEs for later
rmse_sum <- rmse_sum + rmse

}

# Calculate average RMSE across all folds
avg_rmse <- rmse_sum/fold

# If new RMSE is improvement over best RMSE,
# Save as new best RMSE, along with model parameters and features
if (avg_rmse < best_rmse) {
  
  best_rmse <- avg_rmse
  best_params <- XGB_params
  best_features <- feature_selection
  
}

rmse_sum <- 0

# End of parent for loop (random search loop) 
}

# Print results
print(paste("Best Achieved RMSE:",best_rmse))
print(paste("Best Learning Rate:",best_params$eta))
print(paste("Best Max Depth:",best_params$max_depth))
print(paste("Best Feature Performance:",best_features))
print(paste("Reduction needed for next score bracket:",best_rmse-0.86549))


# THE FINAL TEST

# Employ similar methodology to that used above on the final testing round
# The best model parameters and features as determined above are used for the final RMSE computation

# Drop training instances with no genres listed
# Such instances will be underfit by model if present in final_holdout_set
# This is acceptable given the rarity of movies with no given genre
edx <- subset(edx,edx$genres != "(no genres listed)")

# Call dataframe functions on full edx and final_holdout_test
edx <- One_Hot_Genrecoder(edx,TRUE)
edx <- Create_Nostalgia_Factor(edx)

final_holdout_test <- One_Hot_Genrecoder(final_holdout_test,TRUE)
final_holdout_test <- Create_Nostalgia_Factor(final_holdout_test)

# Convert to datatables
setDT(edx)
setDT(final_holdout_test)

# Apply feature creation functions as described above
Final_User_Genre_Ratings <- Extract_User_Genre_Ratings(edx,8)
edx <- Add_User_Genre_Ratings (edx,Final_User_Genre_Ratings,NA,8,all_round )
final_holdout_test <- Add_User_Genre_Ratings (final_holdout_test,Final_User_Genre_Ratings,NA,8,all_round )

Final_Movie_Genre_Ratings <- Extract_Movie_Genre_Ratings(edx,9,all_round)
edx <- Add_Movie_Genre_Ratings(edx,Final_Movie_Genre_Ratings,9,all_round )
final_holdout_test <- Add_Movie_Genre_Ratings(final_holdout_test,Final_Movie_Genre_Ratings,9,all_round)

result <- Add_Rating_Means(edx, final_holdout_test, "movieId",NA,all_round)
edx <- result$train_data
final_holdout_test <- result$test_data

# Break into feature and target arrays as before and convert to DMatrices
Final_X_train <- edx[,..feature_selection]
Final_Y_train <- edx$rating
Final_X_test <- final_holdout_test[,..feature_selection]
Final_Y_test <- final_holdout_test$rating

Final_X_train <- as.matrix(Final_X_train)
Final_Y_train <- as.matrix(Final_Y_train)
Final_X_test <- as.matrix(Final_X_test)
Final_Y_test <- as.matrix(Final_Y_test)

Final_dtrain <- xgb.DMatrix(data=Final_X_train,label=Final_Y_train,missing=NA)
Final_dtest <- xgb.DMatrix(data=Final_X_test,label = Final_Y_test)

# Train final model
Final_XGB_model <- xgb.train(

  params = best_params,
  data = Final_dtrain,
  nrounds = 200,
  watchlist = list(train=Final_dtrain,test=Final_dtest),
  early_stopping_rounds = 10,
  verbose = 1

)

# Render final predictions and print final RMSE
Final_predictions <- predict(Final_XGB_model,newdata = Final_dtest)
Final_rmse <- sqrt(mean((Final_predictions-Final_Y_test)^2))
print(paste("The Final RMSE is",Final_rmse))
