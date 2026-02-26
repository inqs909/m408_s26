library(torch)
library(torchvision)

train_ds <- cifar10_dataset(
  root = "../",
  train = TRUE,
  download = TRUE,
  transform = transform_to_tensor
)

classes <- c("airplane","automobile","bird","cat","deer",
             "dog","frog","horse","ship","truck")

sample <- train_ds[1]
img <- sample[[1]]$permute(c(2,3,1))
label <- sample[[2]]

plot(as.raster(as.array(img)), axes = FALSE,
     main = classes[label + 1])

par(mfrow = c(1,1))


# Extract channels
red   <- img[,,1]
green <- img[,,2]
blue  <- img[,,3]

# Convert to arrays
red   <- as.array(red)
green <- as.array(green)
blue  <- as.array(blue)

make_colored_channel <- function(channel_matrix, color_index) {
  rgb_array <- array(0, dim = c(32, 32, 3))
  rgb_array[,,color_index] <- channel_matrix
  rgb_array
}

plot(as.raster(make_colored_channel(red, 1)),
     axes = FALSE, main = "Red")

plot(as.raster(make_colored_channel(green, 2)),
     axes = FALSE, main = "Green")

plot(as.raster(make_colored_channel(blue, 3)),
     axes = FALSE, main = "Blue")
