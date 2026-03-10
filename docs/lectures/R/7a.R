library(tidyverse)
library(torch)
library(luz)
library(torchvision)
set.seed(909)
torch_manual_seed(909)



# dir <- "./"
dir <- "../"


train_ds <- cifar10_dataset(
  root = dir,
  train = TRUE, 
  download = TRUE, 
  transform = transform_to_tensor
)

test_ds <- cifar10_dataset(
  dir, 
  train = FALSE, 
  download = TRUE,
  transform = transform_to_tensor
)

train_dl <- dataloader(train_ds,
  batch_size = 128,
  shuffle = TRUE
)

## Used to build cnn

b1 <- train_dl |>
  dataloader_make_iter() |> 
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
  },
  forward = function(x) {
    x |> 
      self$conv() |> 
      self$relu() |> 
      self$pool()
  }
)

b1$x$shape
c1 <- conv_block(3, 8)
c2 <- conv_block(8, 16)
c3 <- conv_block(16, 32)
c4 <- conv_block(32, 64)

#l1 <- nn_linear(512, 16)
b1$x |> 
  c1() |> 
  c2() |> 
  c3() |> 
  c4() |> 
  torch_flatten(start_dim = 2) |> 
  (\(x) x$shape)() 


model1 <- nn_module(
  initialize = function() {
    self$conv <- nn_sequential(
      conv_block(3, 8),
      conv_block(8, 16),
      conv_block(16, 32),
      conv_block(32, 64)
    )
    self$output <- nn_sequential(
      nn_dropout(0.5),
      nn_linear(256, 16),
      nn_relu(),
      nn_linear(16, 10)
    )
  },
  forward = function(x) {
    x |> 
      self$conv() |> 
      torch_flatten(start_dim = 2) |> 
      self$output()
  }
)

first <- Sys.time()
fitted1 <- model1 |> 
  setup(
    loss = nn_cross_entropy_loss(),
    optimizer = optim_rmsprop,
    metrics = list(
      luz_metric_accuracy()
    )
  ) |> 
  set_opt_hparams(lr = 0.001) |> 
  fit(
    train_dl,
    epochs = 5
  )
Sys.time() - first

plot(fitted1)

## Batch Normalization

model2 <- nn_module(
  initialize = function() {
    self$conv <- nn_sequential(
      conv_block(3, 8),
      nn_batch_norm2d(8),
      conv_block(8, 16),
      nn_batch_norm2d(16),
      conv_block(16, 32),
      nn_batch_norm2d(32),
      conv_block(32, 64)
    )
    self$output <- nn_sequential(
      nn_dropout(0.5),
      nn_linear(256, 16),
      nn_relu(),
      nn_linear(16, 10)
    )    
  },
  forward = function(x) {
    x |> 
      self$conv() |> 
      torch_flatten(start_dim = 2) |> 
      self$output()
  }
)


first <- Sys.time()
fitted2 <- model2 |> 
  setup(
    loss = nn_cross_entropy_loss(),
    optimizer = optim_rmsprop,
    metrics = list(
      luz_metric_accuracy()
    )
  ) |> 
  set_opt_hparams(lr = 0.001) |> 
  fit(
    train_dl,
    epochs = 5
  )
Sys.time() - first

plot(fitted2)

## Dynamic Learning Rate

model3 <- model2 |> 
  setup(
    loss = nn_cross_entropy_loss(),
    optimizer = optim_rmsprop,
    metrics = list(
      luz_metric_accuracy()
    )
  )

rates <- model3 |> lr_finder(train_dl, start_lr = 0.000001, end_lr = 0.3)
plot(rates)


first <- Sys.time()
fitted3 <- model3 |> 
  fit(train_dl, 
      epochs = 3,
      callbacks = list(
        luz_callback_lr_scheduler(
          lr_one_cycle,
          max_lr = 0.001,
          epochs = 5,
          steps_per_epoch = length(train_dl),
          call_on = "on_batch_end"
        )
      )
  )
Sys.time() - first
plot(fitted3)

## Transfer Learning

resnet <- model_resnet18(pretrained = TRUE)
resnet
resnet$fc

convnet <- nn_module(
  initialize = function() {
    self$model <- model_resnet18(pretrained = TRUE)
    for (par in self$parameters) {
      par$requires_grad_(FALSE)
    }
    self$model$fc <- nn_sequential(
      nn_linear(self$model$fc$in_features, 10)
    )
  },
  forward = function(x) {
    self$model(x)
  }
)

model4 <- convnet |> 
  setup(
    loss = nn_cross_entropy_loss(),
    optimizer = optim_adam,
    metrics = list(
      luz_metric_accuracy()
    )
  ) 


first <- Sys.time()
fitted4 <- model4 |> 
  fit(train_dl, epochs = 5)
Sys.time() - first
