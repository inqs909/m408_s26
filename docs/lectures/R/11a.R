## R Packages

library(torch)
library(luz) # high-level interface for torch
library(torchdatasets) # for datasets we are going to use
library(zeallot)
torch_manual_seed(909)

## Data

set.seed(1)
max_features <- 10000
imdb_train <- imdb_dataset(
  root = "../", 
  download = TRUE,
  split="train",
  num_words = max_features
)
imdb_test <- imdb_dataset(
  root = "../", 
  download = TRUE,
  split="test",
  num_words = max_features
)

## Data

imdb_train[1]$x[1:12]

## Decode

word_index <- imdb_train$vocabulary
decode_review <- function(text, word_index) {
   word <- names(word_index)
   idx <- unlist(word_index, use.names = FALSE)
   word <- c("<PAD>", "<START>", "<UNK>", word)
   words <- word[text]
   paste(words, collapse = " ")
}
decode_review(imdb_train[1]$x[1:12], word_index)

## One-hot Decode

library(Matrix)
one_hot <- function(sequences, dimension) {
   seqlen <- sapply(sequences, length)
   n <- length(seqlen)
   rowind <- rep(1:n, seqlen)
   colind <- unlist(sequences)
   sparseMatrix(i = rowind, j = colind,
      dims = c(n, dimension))
}

# collect all values into a list
train <- seq_along(imdb_train) |> 
  lapply(function(i) imdb_train[i]) |>  
  purrr::transpose()
test <- seq_along(imdb_test) |> 
  lapply(function(i) imdb_test[i]) |> 
  purrr::transpose()

# num_words + padding + start + oov token = 10000 + 3
x_train_1h <- one_hot(train$x, 10000 + 3)
x_test_1h <- one_hot(test$x, 10000 + 3)
dim(x_train_1h)
nnzero(x_train_1h) / (25000 * (10000 + 3))

set.seed(3)
ival <- sample(seq(along = train$y), 2000)
itrain <- seq_along(train$y)[-ival]
y_train <- unlist(train$y)

## Neural Network


## Modules

model <- nn_module(
  initialize = function(input_size = 10000 + 3) {
    self$dense1 <- nn_linear(input_size, 16)
    self$relu <- nn_relu()
    self$dense2 <- nn_linear(16, 16)
    self$output <- nn_linear(16, 1)
  },
  forward = function(x) {
    x |> 
      self$dense1() |>  
      self$relu() |> 
      self$dense2() |>  
      self$relu() |> 
      self$output() |> 
      torch_flatten(start_dim = 1)
  }
)


## Model

model <- model |> 
  setup(
    loss = nn_bce_with_logits_loss(),
    optimizer = optim_rmsprop,
    metrics = list(luz_metric_binary_accuracy_with_logits())
  ) |> 
  set_opt_hparams(lr = 0.001)



## Fit

fitted <- model |> 
  fit(
    # we transform the training and validation data into torch tensors
    list(
      torch_tensor(as.matrix(x_train_1h[itrain,]), dtype = torch_float()), 
      torch_tensor(unlist(train$y[itrain]))
    ),
    valid_data = list(
      torch_tensor(as.matrix(x_train_1h[ival, ]), dtype = torch_float()), 
      torch_tensor(unlist(train$y[ival]))
    ),
    dataloader_options = list(batch_size = 512),
    epochs = 50
  )


  
## Plot


plot(fitted)  



## RNN Document Classification


## Data


maxlen <- 500
num_words <- 10000
imdb_train <- imdb_dataset(root = "../", split = "train", num_words = num_words,
                           maxlen = maxlen)
imdb_test <- imdb_dataset(root = "../", split = "test", num_words = num_words,
                           maxlen = maxlen)

vocab <- c(rep(NA, imdb_train$index_from - 1), imdb_train$get_vocabulary())
tail(names(vocab)[imdb_train[1]$x])

model <- nn_module(
  initialize = function() {
    self$embedding <- nn_embedding(10000 + 3, 32)
    self$lstm <- nn_lstm(input_size = 32, hidden_size = 32, batch_first = TRUE)
    self$dense <- nn_linear(32, 1)
  },
  forward = function(x) {
    c(output, c(hn, cn)) %<-% (x  |>  
      self$embedding() |> 
      self$lstm())
    output[,-1,] |>   # get the last output
      self$dense() |>  
      torch_flatten(start_dim = 1)
  }
)


## Model


model <- model |> 
  setup(
    loss = nn_bce_with_logits_loss(),
    optimizer = optim_rmsprop,
    metrics = list(luz_metric_binary_accuracy_with_logits())
  ) |> 
  set_opt_hparams(lr = 0.001)



## Fit


fitted <- model |> fit(
  imdb_train, 
  epochs = 5,
  dataloader_options = list(batch_size = 128),
  valid_data = imdb_test
)



plot(fitted)



## Evaluation


predy <- torch_sigmoid(predict(fitted, imdb_test)) > 0.5
evaluate(fitted, imdb_test, dataloader_options = list(batch_size = 512))

