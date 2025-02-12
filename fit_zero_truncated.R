# Load required packages
print("Loading VGAM package...")
library(VGAM)

# Check if file exists
data_file <- "data/frequency_data.csv"
print(paste("Checking if file exists:", data_file))
if (!file.exists(data_file)) {
  stop(paste("Error: File not found:", data_file))
}

# Read your data
print("Reading data...")
data <- try(read.csv(data_file))
if (inherits(data, "try-error")) {
  stop(paste("Error reading CSV file:", data_file))
}

# Print data summary
print("Data summary:")
print(str(data))
print(head(data))

# Check column names
print("Column names in dataset:")
print(colnames(data))

# Fit zero-truncated quasi-Poisson model
print("Fitting model...")
tryCatch({
  # Replace these column names with your actual column names from the data
  model <- vglm(n_events ~ revenue + industry + employees, 
                family = pospoisson(),  # for zero-truncated Poisson
                data = data)
  
  print("Model fit successful!")
  
  # Print model summary
  print("Model summary:")
  print(summary(model))
  
  # Get coefficients
  print("Model coefficients:")
  print(coef(model))
  
  # Get fitted values
  print("First few fitted values:")
  print(head(fitted(model)))
  
  # Get residuals
  print("First few residuals:")
  print(head(residuals(model)))
  
}, error = function(e) {
  print(paste("Error in model fitting:", e$message))
})
