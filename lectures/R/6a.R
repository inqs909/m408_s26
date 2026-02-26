library(torch)
library(luz)
library(torchvision)
set.seed(777)
torch_manual_seed(777)
transform <- function(x) {
  transform_to_tensor(x)
}

# dir <- "./"
dir <- "../"


train_ds <- cifar10_dataset(
  root = dir,
  train = TRUE, 
  download = TRUE, 
  transform = transform
)

test_ds <- cifar10_dataset(
  dir, 
  train = FALSE, 
  download = TRUE,
  transform = transform
)

train_dl <- dataloader(train_ds,
  batch_size = 128,
  shuffle = TRUE
)

conv_block <- nn_module(
  initialize = function(in_channels, out_channels) {
    self$conv <- nn_conv2d(
      in_channels = in_channels, 
      out_channels = out_channels, 
      kernel_size = c(3,3), 
      padding = "same"
    )
    self$relu <- nn_relu()
    self$pool <- nn_max_pool2d(kernel_size = c(2,2))
  },
  forward = function(x) {
    x |> 
      self$conv() |> 
      self$relu() |> 
      self$pool()
  }
)

model <- nn_module(
  initialize = function() {
    self$conv <- nn_sequential(
      conv_block(3, 8),
      conv_block(8, 16)
    )
    self$output <- nn_sequential(
      nn_dropout(0.5),
      nn_linear(1024, 16),
      nn_relu(),
      nn_linear(16, 10)
    )
  },
  forward = function(x) {
    x %>% 
      self$conv() %>% 
      torch_flatten(start_dim = 2) %>% 
      self$output()
  }
)

first <- Sys.time()
fitted <- model %>% 
  setup(
    loss = nn_cross_entropy_loss(),
    optimizer = optim_rmsprop, 
    metrics = list(luz_metric_accuracy())
  ) %>% 
  set_opt_hparams(lr = 0.001) %>% 
  fit(
    train_dl,
    epochs = 5
  )
Sys.time() - first

plot(fitted)


## Used to build cnn

b1 <- train_dl |>
  dataloader_make_iter() |> 
  dataloader_next()

b1$x$shape
c1 <- conv_block(3, 8)
c2 <- conv_block(8, 16)
#l1 <- nn_linear(512, 16)
b1$x |> 
  c1() |> 
  c2() |> 
  # c3() |> 
  # c4() |> 
  torch_flatten(start_dim = 2) |> 
  (\(x) x$shape)() 



