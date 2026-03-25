## R Packages

# install.packages("tsibble")
# install.packages("tsibbledata")

library(tidyverse)
library(torch)
library(luz) # high-level interface for torch

torch_manual_seed(909)
library(tsibble)
library(tsibbledata)
vic_elec


demand_dataset <- dataset(
  name = "demand_dataset",
  initialize = function(x,
                        n_timesteps,
                        n_forecast,
                        sample_frac = 1) {
    self$n_timesteps <- n_timesteps
    self$n_forecast <- n_forecast
    self$x <- torch_tensor((x - train_mean) / train_sd)

    n <- length(self$x) -
      self$n_timesteps - self$n_forecast + 1

    self$starts <- sort(sample.int(
      n = n,
      size = n * sample_frac
    ))
  },
  .getitem = function(i) {
    start <- self$starts[i]
    end <- start + self$n_timesteps - 1

    list(
      x = self$x[start:end],
      y = self$x[(end + 1):(end + self$n_forecast)]$
        squeeze(2)
    )
  },
  .length = function() {
    length(self$starts)
  }
)


demand_hourly <- vic_elec |> 
  index_by(Hour = floor_date(Time, "hour")) |> 
  summarise(
    Demand = sum(Demand))

demand_train <- demand_hourly |> 
  filter(year(Hour) == 2012) |> 
  as_tibble() |> 
  select(Demand) |> 
  as.matrix()

demand_valid <- demand_hourly |> 
  filter(year(Hour) == 2013) |> 
  as_tibble() |> 
  select(Demand) |> 
  as.matrix()

demand_test <- demand_hourly |> 
  filter(year(Hour) == 2014) |> 
  as_tibble() |> 
  select(Demand) |> 
  as.matrix()

train_mean <- mean(demand_train)
train_sd <- sd(demand_train)


n_timesteps <- 7 * 24
n_forecast <- 7 * 24

train_ds <- demand_dataset(
  demand_train,
  n_timesteps,
  n_forecast,
  sample_frac = 1
)
valid_ds <- demand_dataset(
  demand_valid,
  n_timesteps,
  n_forecast,
  sample_frac = 1
)
test_ds <- demand_dataset(
  demand_test,
  n_timesteps,
  n_forecast
)

batch_size <- 128
train_dl <- train_ds |> 
  dataloader(batch_size = batch_size, shuffle = TRUE)
valid_dl <- valid_ds |> 
  dataloader(batch_size = batch_size)
test_dl <- test_ds |> 
  dataloader(batch_size = length(test_ds))

b <- train_dl |> 
  dataloader_make_iter() |> 
  dataloader_next()

dim(b$x)
dim(b$y)


input_size <- 1
hidden_size <- 32
num_layers <- 2
rec_dropout <- 0.2

t1 <- nn_lstm(
      input_size = input_size,
      hidden_size = hidden_size,
      num_layers = num_layers,
      dropout = rec_dropout,
      batch_first = TRUE
    )

t1(b[[1]])[[1]][, dim(b[[1]])[2], ] |> dim()

model <- nn_module(
  initialize = function(input_size,
                        hidden_size,
                        linear_size,
                        output_size,
                        dropout = 0.2,
                        num_layers = 1,
                        rec_dropout = 0) {
    self$num_layers <- num_layers

    self$rnn <- nn_lstm(
      input_size = input_size,
      hidden_size = hidden_size,
      num_layers = num_layers,
      dropout = rec_dropout,
      batch_first = TRUE
    )

    self$dropout <- nn_dropout(dropout)
    self$mlp <- nn_sequential(
      nn_linear(hidden_size, linear_size),
      nn_relu(),
      nn_dropout(dropout),
      nn_linear(linear_size, output_size)
    )
  },
  forward = function(x) {
    x <- self$rnn(x)[[2]][[1]][self$num_layers, , ] |> 
      self$mlp()
  }
)
input_size <- 1
hidden_size <- 32
linear_size <- 512
dropout <- 0.5
num_layers <- 2
rec_dropout <- 0.2

model <- model |> 
  setup(optimizer = optim_adam, loss = nn_mse_loss()) |> 
  set_hparams(
    input_size = input_size,
    hidden_size = hidden_size,
    linear_size = linear_size,
    output_size = n_forecast,
    num_layers = num_layers,
    rec_dropout = rec_dropout
  )


fitted <- model |> 
  fit(train_dl, epochs = 30, valid_data = valid_dl,
      verbose = TRUE)

plot(fitted)

evaluate(fitted, test_dl)

demand_viz <- demand_hourly |> 
  filter(year(Hour) == 2014, month(Hour) == 12)

demand_viz_matrix <- demand_viz |> 
  as_tibble() |> 
  select(Demand) |> 
  as.matrix()

n_obs <- nrow(demand_viz_matrix)

viz_ds <- demand_dataset(
  demand_viz_matrix,
  n_timesteps,
  n_forecast
)
viz_dl <- viz_ds |> 
  dataloader(batch_size = length(viz_ds))

preds <- predict(fitted, viz_dl)
preds <- preds$to(device = "cpu") |> 
  as.matrix()

example_preds <- vector(mode = "list", length = 3)
example_indices <- c(1, 201, 401)

for (i in seq_along(example_indices)) {
  cur_obs <- example_indices[i]
  example_preds[[i]] <- c(
    rep(NA, n_timesteps + cur_obs - 1),
    preds[cur_obs, ],
    rep(
      NA,
      n_obs - cur_obs + 1 - n_timesteps - n_forecast
    )
  )
}

pred_ts <- demand_viz |> 
  select(Demand) |> 
  add_column(
    p1 = example_preds[[1]] * train_sd + train_mean,
    p2 = example_preds[[2]] * train_sd + train_mean,
    p3 = example_preds[[3]] * train_sd + train_mean) |> 
  pivot_longer(-Hour) |> 
  update_tsibble(key = name)

pred_ts |> 
    ggplot(aes(Hour, value, color = name)) +
      geom_line() +
  scale_colour_manual(
    values = c(
      "#08c5d1", "#00353f", "#ffbf66", "#d46f4d"
    )
  ) +
  theme_minimal() +
  theme(legend.position = "None")
