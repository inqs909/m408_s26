library(torch)
library(luz)
library(torchvision)
set.seed(777)
torch_manual_seed(777)
transform <- function(x) {
  transform_to_tensor(x)
}

train_ds <- cifar10_dataset(
  root = "./", 
  train = TRUE, 
  download = TRUE, 
  transform = transform
)

test_ds <- cifar10_dataset(
  root = "./", 
  train = FALSE, 
  download = TRUE,
  transform = transform
)

train_dl <- dataloader(train_ds,
  batch_size = 128,
  shuffle = TRUE
)
batch <- train_dl %>%
  dataloader_make_iter() %>%
  dataloader_next()

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
    self$batch <- nn_batch_norm2d(out_channels)
  },
  forward = function(x) {
    x |> 
      self$conv() |> 
      self$relu() |> 
      self$batch() |> 
      self$pool()
  }
)

model <- nn_module(
  initialize = function() {
    self$conv <- nn_sequential(
      conv_block(3, 4),
      conv_block(4, 8),
      conv_block(8, 16),
      conv_block(16, 32)
    )
    self$output <- nn_sequential(
      nn_dropout(0.5),
      nn_linear(2*2*32, 16),
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
model()
Sys.time()
fitted <- model %>% 
  setup(
    loss = nn_cross_entropy_loss(),
    optimizer = optim_rmsprop, 
    metrics = list(luz_metric_accuracy())
  ) %>% 
  set_opt_hparams(lr = 0.001) %>% 
  fit(
    train_ds,
    epochs = 5, #10,
    valid_data = 0.2,
    dataloader_options = list(batch_size = 128)
  )
Sys.time()
