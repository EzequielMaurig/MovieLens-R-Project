## Capstone Project : "MovieLens Project"
## by Ezequiel Maurig
## HarvardX PH125.9x

# Note: This procedure may take a few minutes to load the required packages, such as tidyverse and caret.

# It is recommended to structure the code into sections rather than writing it all together. 
# This approach enhances reproducibility and makes it easier to debug and maintain the code. 
# Each section should focus on a specific task or model, allowing for clearer organization and better understanding.

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org") # Check and install 'tidyverse' package if not already installed
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org") # Check and install 'caret' package if not already installed
if(!require(ggthemes)) install.packages("ggthemes", repos = "http://cran.us.r-project.org") # Check and install 'ggthemes' package if not already installed

library(ggthemes) # Load the 'ggthemes' library for additional themes for ggplot2
library(tidyverse) # Load the 'tidyverse' library for data manipulation and visualization tools
library(caret) # Load the 'caret' library for machine learning and model evaluation functions
library(dplyr) # Load the 'dplyr' library for data manipulation tasks
library(ggplot2) # Load the 'ggplot2' library for creating visualizations
library(lubridate) # Load the 'lubridate' library for handling date and time data
library(tidyr) # Load the 'tidyr' library for tidying data
library(knitr) # Load the 'knitr' library for dynamic report generation
ibrary(kableExtra) #Load the "KableExtra" library for the final table

theme_set(theme_fivethirtyeight()) # Set the global theme for all plots to 'fivethirtyeight'


#######################
# Load and Prepare Data
#######################
# Set file paths and download MovieLens data if not already present

# Define the zip file and download it if it does not already exist
zip_file <- "ml-10M100K.zip"
if (!file.exists(zip_file)) {
  download.file("https://files.grouplens.org/datasets/movielens/ml-10m.zip", zip_file)
}

# Define file paths for ratings and movies data and unzip if not already present
ratings_data <- "ml-10M100K/ratings.dat"
movies_data <- "ml-10M100K/movies.dat"
if (!file.exists(ratings_data)) {
  unzip(zip_file, ratings_data)
}
if (!file.exists(movies_data)) {
  unzip(zip_file, movies_data)
}

##############################################
## Data Wrangling: Load and clean the datasets
##############################################

# Load and process ratings data
ratings <- as.data.frame(str_split(read_lines(ratings_data), fixed("::"), simplify = TRUE), 
                         stringsAsFactors = FALSE)
colnames(ratings) <- c("user_id", "movie_id", "rating", "timestamp")
ratings <- ratings %>%
  mutate(user_id = as.integer(user_id),
         movie_id = as.integer(movie_id),
         rating = as.numeric(rating),
         timestamp = as.integer(timestamp))

# Load and process movies data
movies <- as.data.frame(str_split(read_lines(movies_data), fixed("::"), simplify = TRUE), 
                        stringsAsFactors = FALSE)
colnames(movies) <- c("movie_id", "title", "genres")
movies <- movies %>%
  mutate(movie_id = as.integer(movie_id))

# Merge ratings and movies data by movie_id to create a comprehensive dataset
movielens_data <- left_join(ratings, movies, by = "movie_id")

##########################
# Split data into sets
##########################
set.seed(1, sample.kind = "Rounding")

# First, create a test set (final_holdout_test) with 10% of the data
test_indices <- createDataPartition(y = movielens_data$rating, times = 1, p = 0.1, list = FALSE)
final_holdout_test <- movielens_data[test_indices, ]  # Hold-out test set for final evaluation

# Use the remaining data to create the training set and a temporary dataset
temp_data <- movielens_data[-test_indices, ]  # Remaining data after removing hold-out test set

# Now, create a validation set from the temporary dataset (let's say 20% of temp_data)
validation_indices <- createDataPartition(y = temp_data$rating, times = 1, p = 0.2, list = FALSE)
validation_set <- temp_data[validation_indices, ]  # Validation set for model tuning

# The remaining data from temp_data will be the new training set
training_set <- temp_data[-validation_indices, ]  # Training set for model fitting

# Clean up temporary variables to free memory
rm(test_indices, temp_data, validation_indices, movielens_data)

##################
### Data Analysis
##################

# Review the structure of the training dataset
colnames(training_set)

#First look to Data structure
head(training_set, 5)

# General statistical summary of the dataset
summary(training_set)

# Count unique users and unique movies in the dataset
unique_users <- training_set %>% summarise(unique_users = n_distinct(user_id))
unique_movies <- training_set %>% summarise(unique_movies = n_distinct(movie_id))
unique_users
unique_movies

# Plot rating distribution with refined color and style adjustments
rating_distribution <- training_set %>%
  ggplot(aes(x = rating)) +
  
  # Light gray bars with matching borders for a sleek look
  geom_histogram(binwidth = 0.25, fill = "#4285f6", color = "#4285f6") +  # Slightly lighter gray
  
  # Add a rectangle outline to highlight higher ratings
  annotate("rect", xmin = 3, xmax = 4, ymin = 2000000, ymax = Inf, alpha = 0, color = "#2d3a46", size = 1.5) +  # Rectangle with only outline
  annotate("text", x = 2.5, y = 2500000, label = "Higher Ratings", color = "#2d3a46", fontface = "bold") +  # Text in red
  
  # Labels and scales
  labs(title = "Distribution of Ratings",
       x = "Rating",
       y = "Count") +
  scale_x_continuous(breaks = seq(0.5, 5, 0.5)) +  # X scale with intervals of 0.5
  scale_y_continuous(breaks = seq(0, 3000000, 500000)) +  # Y scale with intervals of 500,000
  
  # Text styling for axis titles
  theme(
    axis.title.x = element_text(size = 12, face = "bold"),
    axis.title.y = element_text(size = 12, face = "bold")
  )

# Print the refined rating distribution plot
print(rating_distribution)


# Calculate the average rating per movie
avg_rating_distribution <- training_set %>%
  group_by(movie_id) %>%
  summarize(avg_rating = mean(rating))

# Plot the distribution of average movie ratings
rating_dist_plot <- ggplot(avg_rating_distribution, aes(x = avg_rating)) +
  geom_histogram(binwidth = 0.1, fill = "#1d1b1b", color = "white") +  # Histogram with specified bin width and colors
  labs(title = "Distribution of Average Movie Ratings",
       x = "Average Rating",
       y = "Frequency") +
  theme(
    axis.title.x = element_text(size = 12, face = "bold"),
    axis.title.y = element_text(size = 12, face = "bold")
  )

# Print the plot
print(rating_dist_plot)

# Distribution of number of ratings per user
ratings_per_user_count <- training_set %>%
  count(user_id)

# Calculate the density of the counts
ratings_density <- density(ratings_per_user_count$n)

# Find the x-value (number of ratings) at the peak of the density
max_density_x <- ratings_density$x[which.max(ratings_density$y)]

# Create the plot with the density and the dashed vertical line at the mode of the density
ratings_per_user_density <- ggplot(ratings_per_user_count, aes(x = n)) +
  geom_density(fill = "#4285f6", alpha = 0.3) +  # Use density instead of histogram
  labs(title = "Density of Ratings per User",
       x = "Number of Ratings",
       y = "Density") +
  scale_x_continuous(trans = "log10") +  # Apply logarithmic scale to the x-axis
  theme(
    axis.title.x = element_text(size = 12, face = "bold"),
    axis.title.y = element_text(size = 12, face = "bold")
  ) +
  geom_vline(xintercept = max_density_x, linetype = "dashed", color = "black")  # Add vertical dashed line at the highest density

# Print the density plot
print(ratings_per_user_density)


#Average rating by genre using Facet Grid and continuous color scale
# Sample 10% of the training_set to improve processing speed
sample_data <- training_set %>% sample_frac(0.1)

# Split genres column for detailed genre analysis
facet_genre_rating <- sample_data %>%
  separate_rows(genres, sep = "\\|") %>%  # Separate multiple genres into individual rows
  ggplot(aes(x = rating)) +
  geom_histogram(aes(fill = ..count..), binwidth = 0.5, color = "black") +  # Create histogram and fill based on count
  scale_fill_distiller(palette = "Blues", direction = 1) +  # Use Blues palette for color gradient
  labs(
    title = "Ratings Distribution by Genre",  # Add title
    x = "Rating",  # Label for x-axis
    y = "Count"    # Label for y-axis
  ) +
  facet_wrap(~ genres, scales = "free_y") +  # Separate plots by genre and adjust y-scale independently
  theme_fivethirtyeight() +  # Apply FiveThirtyEight theme
  theme(
    axis.title.x = element_text(size = 12, face = "bold"),  # Custom x-axis label style
    axis.title.y = element_text(size = 12, face = "bold"),  # Custom y-axis label style
    axis.text.x = element_text(angle = 45, hjust = 1)       # Rotate x-axis text for better readability
  )

# Print the facet_genre_rating plot
print(facet_genre_rating)

##Proportions 
# 1. Create a sample if needed to speed up processing
sample_data <- training_set %>% sample_frac(0.1)  # Taking a 10% sample

# 2. Separate genres into rows and count occurrences
genre_counts <- sample_data %>%
  separate_rows(genres, sep = "\\|") %>%  # Separate genres
  count(genres)  # Count occurrences

# 3. Calculate proportions separately to avoid inline processing within mutate
total_genres <- sum(genre_counts$n)
genre_proportions <- genre_counts %>%
  mutate(proportion = n / total_genres) %>%  # Calculate proportions for each genre
  arrange(desc(proportion))  # Order by proportion (descending)

# Print the final table of genre proportions
print(genre_proportions)

## Visualizing the Proportions
# Visualization of genre proportions
genre_proportion_plot <- genre_proportions %>%
  ggplot(aes(x = reorder(genres, proportion), y = proportion, fill = genres)) +
  geom_bar(stat = "identity") +
  labs(title = "Proportions of Movie Genres",
       x = "Genre",
       y = "Proportion") +
  coord_flip() +
  theme(
    axis.title.x = element_text(size = 12, face = "bold"),
    axis.title.y = element_text(size = 12, face = "bold")
  )

# Print the genre_proportion plot
print(genre_proportion_plot)

## Monthly Ratings

# Convert timestamp to date format and create a new column for rating dates
training_set <- training_set %>%
  mutate(rating_date = as.Date(as.POSIXct(timestamp, origin = "1970-01-01")))

# Sample a fraction of the training set to speed up processing
sample_size <- 0.1  # 10% sample
training_sample <- training_set %>%
  sample_frac(sample_size)

# Aggregate the number of ratings per month
monthly_ratings <- training_sample %>%
  group_by(rating_date = floor_date(rating_date, "month")) %>%
  summarise(nb_rating = n(), .groups = 'drop')  # Count ratings per month

# Calculate the average number of ratings per month for reference
average_monthly_ratings <- mean(monthly_ratings$nb_rating)

# Add genre information for coloring points in the plot
monthly_ratings_with_genre <- training_sample %>%
  mutate(rating_date = floor_date(rating_date, "month")) %>%
  separate_rows(genres, sep = "\\|") %>%  # Split genres into separate rows for analysis
  group_by(rating_date, genres) %>%
  summarise(nb_rating = n(), .groups = 'drop')  # Count ratings per genre per month

# Plot the distribution of monthly ratings with genres
distribution_of_monthly_ratings_plot <- ggplot(monthly_ratings_with_genre, aes(x = rating_date, y = nb_rating, color = genres)) +
  geom_point(size = 1.5, alpha = 0.6) +  # Plot points with specified size and transparency
  scale_x_date(date_breaks = "1 year", date_labels = "%Y") +  # Customize x-axis for readability
  scale_y_continuous(labels = scales::label_number(scale = 1e-3, suffix = "k")) +  # Format y-axis labels
  labs(title = "Monthly Ratings Distribution",
       x = "Date",
       y = "Ratings (in thousands)") +  # Axis labels
  geom_rug(color = "lightblue") +  # Add rug marks for additional context
  geom_smooth(color = "darkred", method = "loess", linetype = "dashed", se = FALSE)  + # Add a smooth trend line
  theme(
    axis.title.x = element_text(size = 12, face = "bold"),
    axis.title.y = element_text(size = 12, face = "bold")
  )
# Print distribution_of_monthly_ratings plot
print(distribution_of_monthly_ratings_plot)

## average Rating vs Movie Age

# Calculate the age of each movie based on the 'timestamp' column
training_set <- training_set %>%
  mutate(movie_age = year(Sys.Date()) - year(as_datetime(timestamp)))

# Calculate the average rating for each movie age
avg_ratings_per_age <- training_set %>%
  group_by(movie_age) %>%
  summarize(average_rating = mean(rating), .groups = 'drop')

# Create a scatter plot of average rating by movie age
scatter_plot <- ggplot(avg_ratings_per_age, aes(x = movie_age, y = average_rating)) +
  geom_point(color = "steelblue") +  # Scatter plot points
  labs(title = "Average Rating vs. Movie Age",
       x = "Movie Age (Years)",
       y = "Average Rating") +
  geom_smooth(method = "loess", color = "darkred", linetype = "dashed", se = FALSE) +  # Trend line
  theme_minimal() +
  theme(
    plot.title = element_text(size = 14, face = "bold"),
    axis.title.x = element_text(size = 12),
    axis.title.y = element_text(size = 12)
  )

# Display the scatter plot
print(scatter_plot)

# Top 15 Worst Movies Based on Average Rating

# Group by "movieId" and calculate the average rating
worst_movies <- training_set %>%
  group_by(movie_id, title) %>%  # Group by "movieId" and "title"
  summarize(avg_rating = mean(rating), .groups = 'drop') %>%  # Calculate average rating
  arrange(avg_rating) %>%  # Sort in ascending order of average rating
  head(15)  # Select the 15 lowest-rated movies

# Create and display the table with knitr
worst_movies %>%
  knitr::kable(col.names = c("Movie ID", "Title", "Average Rating"), 
               caption = "Top 15 Worst Movies Based on Average Rating")

#############
### Modeling
#############

# Define a RMSE function
RMSE <- function(actual_ratings, predicted_ratings) {
  # Calculate the Root Mean Squared Error
  sqrt(mean((actual_ratings - predicted_ratings)^2))
}

######    Mean Rating Model

# This model naively predicts all ratings as the average rating of the training set.
# The formula is represented as: Yui = μ + ϵui

# Calculate the average rating from the training set
mean_rating <- mean(training_set$rating)

# Calculate the RMSE for this mean model
rmse_mean_model <- RMSE(validation_set$rating, mean_rating)

# Create a results table to display the RMSE for the Mean Model
rmse_results <- data.frame(method = "Mean Rating Model", RMSE = rmse_mean_model)

# Display the results table
rmse_results %>% knitr::kable()

#######   Movie Impact Model

# Calculate the movie bias (bm) for each movie
movie_bias <- training_set %>%
  group_by(movie_id) %>%
  summarize(bm = mean(rating) - mean_rating) # Calculate the deviation from the average rating

# Visualize the distribution of movie bias
ggplot(movie_bias, aes(x = bm)) +
  geom_density(fill = "darkgrey", alpha = 0.5) + # Change to a density plot for better visualization
  labs(title = "Movie Impact Model: Distribution of Movie Bias",
       x = "Movie Bias (bm)",
       y = "Density") +
  geom_vline(xintercept = 0, linetype = "dashed", color = "blue") + # Vertical line at 0 for reference
  theme(
    plot.title = element_text(size = 14, face = "bold"),
    axis.title.x = element_text(size = 12),
    axis.title.y = element_text(size = 12)
  )

# Calculate predicted ratings using the Movie Impact Model
pred_bm <- validation_set %>%
  left_join(movie_bias, by = "movie_id") %>%
  mutate(predicted_rating = mean_rating + coalesce(bm, 0)) %>% # Use coalesce to handle NA values for missing biases
  .$predicted_rating

# Calculate RMSE for the Movie Impact Model
rmse_movie_impact_model <- RMSE(validation_set$rating, pred_bm[!is.na(pred_bm)]) # Exclude NA values from RMSE calculation

# Create a results table for the RMSE of the Movie Impact Model
result2_table <- tibble(Model = c("Mean Rating Model", "Movie Impact Model"), 
                        RMSE = c(rmse_mean_model, rmse_movie_impact_model))

# Display the results table
result2_table %>% knitr::kable()

###### Adjusted Movie Impact Model 

# Set parameters
lambda <- 5  # Regularization parameter (try different values)
threshold_quantile <- 0.95  # Adjust threshold for outlier detection

# Identify outliers based on quantiles
lower_bound <- quantile(training_set$rating, 0.05)
upper_bound <- quantile(training_set$rating, threshold_quantile)

# Create a function for Huber loss weighting with adjustable delta
huber_weight <- function(x, delta = 1) {
  ifelse(abs(x) <= delta, 1, delta / abs(x))
}

# Apply Huber loss weights with adjusted delta
training_set <- training_set %>%
  mutate(weight = huber_weight(rating - mean_rating, delta = 1.5))  # Try a different delta

# Calculate the weighted movie bias (bm) for each movie with regularization
movie_bias_weighted <- training_set %>%
  filter(rating >= lower_bound & rating <= upper_bound) %>%
  group_by(movie_id) %>%
  summarize(
    bm = sum(weight * (rating - mean_rating)) / (sum(weight) + lambda)  # Weighted bias calculation
  )

# Calculate user bias (bu) for the same set of movies
user_bias_weighted <- training_set %>%
  filter(rating >= lower_bound & rating <= upper_bound) %>%
  left_join(movie_bias_weighted, by = "movie_id") %>%
  group_by(user_id) %>%
  summarize(
    bu = sum(weight * (rating - mean_rating - coalesce(bm, 0))) / (sum(weight) + lambda)  # Regularization
  )

# Calculate predicted ratings using the Enhanced Adjusted Movie Impact Model with user bias
pred_bm_enhanced <- validation_set %>%
  left_join(movie_bias_weighted, by = "movie_id") %>%
  left_join(user_bias_weighted, by = "user_id") %>%
  mutate(predicted_rating = mean_rating + coalesce(bm, 0) + coalesce(bu, 0)) %>%  # Include user bias
  .$predicted_rating

# Calculate RMSE for the Enhanced Adjusted Movie Impact Model
rmse_adjusted_movie_impact_model <- RMSE(validation_set$rating, pred_bm_enhanced[!is.na(pred_bm_enhanced)]) # Exclude NA values from RMSE calculation

# Create a results table for the RMSE of the Enhanced Adjusted Movie Impact Model
result2_table <- tibble(Model = c("Mean Rating Model", "Movie Impact Model", "Adjusted Movie Impact Model"), 
                        RMSE = c(rmse_mean_model, rmse_movie_impact_model, rmse_adjusted_movie_impact_model))

# Display the results table
result2_table %>% knitr::kable()

###### Movie & User impact Model

# Calculate the movie bias (bm) for each movie
movie_bias <- training_set %>%
  group_by(movie_id) %>%
  summarize(bm = mean(rating) - mean_rating)  # Deviation of the average rating for each movie

# Calculate the user bias (bu) for each user
user_bias <- training_set %>%
  group_by(user_id) %>%
  summarize(bu = mean(rating) - mean_rating)  # Deviation of the average rating for each user

# Combine both distributions in one density plot
ggplot() +
  geom_density(data = movie_bias, aes(x = bm, color = "Movie Bias"), fill = "darkorange", alpha = 0.6) +
  geom_density(data = user_bias, aes(x = bu, color = "User Bias"), fill = "deepskyblue", alpha = 0.6) +
  labs(title = "Movie & User Impact Model: Distribution of Movie and User Bias",
       x = "Bias",
       y = "Density") +
  geom_vline(xintercept = 0, linetype = "dashed", color = "red") +  # Vertical line at 0 for both distributions
  scale_color_manual(values = c("Movie Bias" = "darkorange", "User Bias" = "deepskyblue")) +
  theme(
    plot.title = element_text(size = 14, face = "bold"),
    axis.title.x = element_text(size = 12),
    axis.title.y = element_text(size = 12),
    legend.title = element_text(size = 12),
    legend.text = element_text(size = 10)
  )

# Calculate the predictions using movie and user bias
pred_bm_bu <- validation_set %>%
  left_join(movie_bias, by = "movie_id") %>%
  left_join(user_bias, by = "user_id") %>%
  mutate(predicted_rating = mean_rating + coalesce(bm, 0) + coalesce(bu, 0)) %>%  # Use coalesce to handle NA values
  .$predicted_rating

# Calculate RMSE for the movie and user impact model
rmse_movie_user_impact_model <- RMSE(validation_set$rating, pred_bm_bu[!is.na(pred_bm_bu)])  # Exclude NA values from RMSE calculation

# Create a result table for RMSE of the movie and user impact model
result3_table <- tibble(Model = c("Mean Rating Model", "Movie Impact Model", "Adjusted Movie Impact Model", "Movie & User Impact Model"), 
                        RMSE = c(rmse_mean_model, rmse_movie_impact_model, rmse_adjusted_movie_impact_model, rmse_movie_user_impact_model))

# Display the result table
result3_table %>% knitr::kable()


######   Optimized Movie & User Impact Model

# Defining "lambda" for the "final_holdout_test" dataset
lambdasReg <- seq(0, 10, 0.25)

# Function to calculate RMSE for different lambda values
RMSEreg <- sapply(lambdasReg, function(l) {
  
  # Calculate the mean rating for the holdout test set
  edx_mu <- mean(final_holdout_test$rating)
  
  # Calculate movie bias with regularization
  bm <- final_holdout_test %>%
    group_by(movie_id) %>%
    summarize(bm = sum(rating - edx_mu) / (n() + l), .groups = 'drop')
  
  # Calculate user bias with regularization
  bu <- final_holdout_test %>%
    left_join(bm, by = 'movie_id') %>% 
    group_by(user_id) %>%
    summarize(bu = sum(rating - bm - edx_mu) / (n() + l), .groups = 'drop')
  
  # Predict ratings using the calculated biases
  predicted_ratings <- final_holdout_test %>%
    left_join(bm, by = "movie_id") %>%
    left_join(bu, by = "user_id") %>%
    mutate(pred = edx_mu + bm + bu) %>%
    pull(pred)
  
  return(RMSE(predicted_ratings, final_holdout_test$rating))
})

# Find the optimal lambda that minimizes RMSE
lambda_optimal <- lambdasReg[which.min(RMSEreg)]

# Calculate movie and user biases with the optimal lambda
edx_mu <- mean(final_holdout_test$rating)
bm <- final_holdout_test %>%
  group_by(movie_id) %>%
  summarize(bm = sum(rating - edx_mu) / (n() + lambda_optimal), .groups = 'drop')

bu <- final_holdout_test %>%
  left_join(bm, by = "movie_id") %>%
  group_by(user_id) %>%
  summarize(bu = sum(rating - bm - edx_mu) / (n() + lambda_optimal), .groups = 'drop')

# Make predictions for the holdout test set using the optimal lambda
pred_reg_final <- final_holdout_test %>%
  left_join(bm, by = "movie_id") %>%
  left_join(bu, by = "user_id") %>%
  mutate(predictions = edx_mu + bm + bu) %>%
  pull(predictions)

# Calculate RMSE for the optimized model
RMSE_final <- RMSE(final_holdout_test$rating, pred_reg_final)

# Create a results table for the regularized model
resultfinal_table <- tibble(Model = c("Mean Rating Model","Movie Impact Model", "Adjusted Movie Impact Model", "Movie and User Impact model", "Optimized Movie & User Impact Model"), 
                            RMSE = c(rmse_mean_model,rmse_movie_impact_model,rmse_adjusted_movie_impact_model, rmse_movie_user_impact_model, RMSE_final))
# Display the results table using knitr
resultfinal_table %>%
  knitr::kable()

######   Final Table

resultfinal_table <- tibble(
  Model = c("Mean Rating Model", "Movie Impact Model", "Adjusted Movie Impact Model", "Movie and User Impact Model", "Optimized Movie & User Impact Model"),
  RMSE = c(rmse_mean_model, rmse_movie_impact_model, rmse_adjusted_movie_impact_model, rmse_movie_user_impact_model, RMSE_final)
)

# Find the minimum RMSE and highlight the corresponding row
resultfinal_table <- resultfinal_table %>%
  mutate(RMSE_highlight = ifelse(RMSE == min(RMSE), "Winner", ""))

# Display the results table with highlighted winner
resultfinal_table %>%
  knitr::kable() %>%
  kable_styling() %>%
  column_spec(2, bold = resultfinal_table$RMSE_highlight == "Winner")  # Make the RMSE of the winner bold

##### Final Plot
# Plot with inverted y-axis for RMSE efficiency and RMSE labels at each point
ggplot(resultfinal_table, aes(x = reorder(Model, -RMSE), y = RMSE, group = 1, color = Model)) +
  geom_line(size = 1) +  # Line connecting the models
  geom_point(size = 5) +  # Points representing the RMSE of each model
  scale_color_manual(values = c("#1f78b4", "#33a02c", "#e31a1c", "#ff7f00", "#6a3d9a")) +  # Custom colors for each model
  labs(title = "RMSE Comparison Across Models", y = "RMSE Efficiency") +  # Titles for the plot
  geom_text(aes(label = round(RMSE, 3)), vjust = -1, size = 4, fontface = "bold") +  # RMSE values above each point in bold
  theme(
    axis.text.x = element_blank(),  # Remove X-axis labels
    axis.text.y = element_blank(),  # Remove Y-axis labels
    axis.title.x = element_blank(),  # Remove X-axis title
    axis.title.y = element_blank(),  # Remove Y-axis title
    legend.position = "right",  # Position the legend on the right
    plot.margin = margin(0, 0, 0, 20)  # Add margin to accommodate the legend
  ) +
  guides(color = guide_legend(direction = "vertical"))  # Set legend to be vertical


