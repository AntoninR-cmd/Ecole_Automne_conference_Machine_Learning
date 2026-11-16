# .........................................................
# Workhorses and domestic functions for the 
# "CNN vs GM" paper
#
# This script defines the two paths described 
# in the Figure1 of the paper. Both rely 
# tibbles provided in the /rda folder as .rda files.
# Yet ad-hoc, they should be really easy to reproduce
# to run the same functions or to adapt the latter.
#
# Author(s): 
#   - Vincent Bonhomme <bonhomme.vincent@gmail.com>
#  Date:
# - 2019 initial version
# - 2024 submission to PCI
# - 2025 resubmission to PCA
# .........................................................

# vgg19_this() implements the full CNN pipe using the
# application vgg19. Hyperparameters are quite vanilla
# and, associated with callbacks, should allow a seamingless
# training.
# - df is a tibble as those found in /rda
# - data_col allows to pick VL/VD columns
# - N_model and N_external allow to change the number
# of files in each partition
# - epochs defines the maximal number of epochs
# - seed is the integer defining the random seed initialization
# to ensure reproducibility
vgg19_this <- function(df, data_col,
                       N_model, N_external=100,
                       # only for the sake of debugging
                       epochs=200, seed=2329){
  
  # deduce a sensible name from the df name
  df_name <- deparse(substitute(df))
  # a cheap clock
  begin_time <- Sys.time()
  # make it internal arguments cause we wont need them
  prop_train=4/5
  # the imput shape that is expected
  input_shape = c(120, 90, 3)
  # our datagenerator simply rescales here
  all_datagen = image_data_generator(rescale=1/255)
  
  # calculate (sub)sample sizes
  N_train <- round(N_model*prop_train)
  N_test  <- N_model - N_train
  N_total <- N_train + N_test + N_external
  
  # test if groupwise sample sizes are ok
  if(any(table(df$status)<N_total)){
    cat("needed ", N_total, "\n available:\n")
    print(table(df$status))
    stop()
  }
  
  # create the (still ordered) vector of labs
  labs <- c(rep("train", N_train),
            rep("test", N_test),
            rep("external", N_external))
  
  coo_col <- str_replace(data_col, "_deep", "_coo")
  
  # for the sake of replicability
  set.seed(seed)
  # now we shuffle
  shuffled_labs <- sample(labs)
  # and add it to df
  master_df <- df %>% 
    group_by(status) %>% 
    sample_n(N_total) %>% 
    mutate(partition=shuffled_labs) %>% 
    ungroup() %>% 
    select(partition, 
           class=status, 
           path=data_col,
           coo=all_of(coo_col))
  
  # now split on partition
  dfs <- master_df %>% select(-coo) %>% split(.$partition)
  
  #### keras part
  # domestic function to flow
  .flow_this <- function(x) {
    flow_images_from_dataframe(
      x,
      directory = "",
      x_col = "path",
      y_col = "class",
      generator=all_datagen,
      target_size = input_shape[1:2],
      class_mode = "binary",
      batch_size = 16,
      shuffle = TRUE,
      seed = seed)
  }
  
  # build the three generators
  train_generator    <- .flow_this(dfs$train)
  test_generator     <- .flow_this(dfs$test)
  external_generator <- .flow_this(dfs$external)
  
  # redefine vgg19 with frozon last layer
  base_model <- application_vgg19(weights = 'imagenet',
                                  include_top = FALSE,
                                  input_shape = input_shape,
                                  classes=2)
  # on top of top we define last layers for our binary prediction
  predictions <- base_model$output %>%
    layer_global_average_pooling_2d() %>%
    layer_dense(units=256, activation="relu") %>%
    layer_dense(units=1, activation="sigmoid")
  # combine the two
  model <- keras_model(inputs=base_model$input, outputs=predictions)
  # and freeze the base model
  freeze_weights(base_model)
  
  # finally compile the model
  model %>%
    compile(optimizer = optimizer_rmsprop(learning_rate=0.01),
            loss = 'binary_crossentropy',
            metrics = c('accuracy'))
  
  # define a nice early stopping and adaptive learning rate
  callbacks_list <- list(
    # is accuracy doesnt improve for 20 epochs, then stop
    callback_early_stopping(
      monitor="accuracy",
      patience=20
    ),
    # if loess on the validation doesnt improve for 10 epochs, 
    # then divide it by 10 until it reaches 1e-7
    callback_reduce_lr_on_plateau(
      monitor="val_loss",
      factor=0.1,
      patience=10,
      min_lr = 1e-7
    )
  )
  
  # in case N_model=50, run at least on 1
  min_val <- floor(N_test/16)
  if (min_val==0) min_val <- 1
  
  # train the model on the new data
  model %>% fit_generator(
    train_generator,
    steps_per_epoch = floor(N_train/16),
    epochs = epochs, # since we have an early stopping we can afford this
    validation_data = test_generator,
    validation_steps = min_val,
    callbacks = callbacks_list
    # callbacks = callback_tensorboard("logs/run_a")
  ) -> model_history
  
  # evaluate on external data
  external_accuracy <- model %>% evaluate_generator(external_generator, steps = floor(N_external/16))
  
  # return this beauty
  list(df=df,
       df_name=df_name,
       master_df=master_df,
       model=model,
       model_history=model_history,
       external_accuracy=external_accuracy,
       duration=Sys.time() - begin_time,
       N_model=N_model,
       epochs=epochs,
       seed=seed,
       data_col=data_col)
}

# this implement the prediction using a vanilla EFT pipe
# using 6 harmonics.
# it accepts the results of vgg19_this to use the very same files
# not the db directly !
# it also mostly relies on builtin Momocs functions
eft6_this <- function(x){
  # build an Out
  coo <- x$master_df$coo
  fac <- x$master_df %>% select(-coo)
  Out(coo, fac) %>% 
    # align outlines
    coo_center() %>% coo_scale() %>% coo_align() %>% 
    coo_slidedirection("right") %>% 
    # EFT
    efourier(6, norm=FALSE) -> al
  # train an LDA on the model
  mod <- al %>% filter(partition != "external") %>% LDA(~class)
  # and use it to predict on external partition
  ext <- al %>% filter(partition == "external")
  re  <- reLDA(ext, mod)
  # extracts the accuracy
  cv <- mean(re$class == ext$fac$class)
  # return the full cross-validation matrix and the accuracy
  list(acc_1_out=mod$CV.correct,
       acc_ext=cv)
}

# this function mostly rearranges results in a more sensible way
clean_this <- function(x){
  tibble(data=x$data,
         view=str_remove(x$view, "_deep"),
         N=x$N_model,
         eft_1out=x$eft$acc_1_out,
         eft_external=x$eft$acc_ext,
         deep_external_acc=x$deep$external_accuracy[2],
         deep_duration=as.numeric(x$deep$duration),
         deep_epochs=length(x$deep$model_history$metrics$loss),
         deep_train=x$deep$N_model,
         deep_external_N=200,
         deep_external_loss=x$deep$external_accuracy[1],
         deep_history=list(x$deep$model_history$metrics),
         seed=x$seed,
         timestamp=Sys.time()) -> out
  print(out)
  out
}

# now we simply combine the two functions
vgg_and_eft_this <- function(df, df_name, data_col, N_model, epochs, seed){
  if (missing(df_name))
    data <- deparse(substitute(df))
  else
    data <- df_name
  # first VGG19
  deep <- vgg19_this(df = df,
                     data_col=data_col,
                     N_model=N_model,
                     epochs = epochs,
                     seed=seed)
  # then EFT on the very same partitions
  eft <- deep %>% eft6_this()
  # returns all intermediate results 
  list(data=data,
       N_model=N_model,
       seed=seed,
       deep=deep, eft=eft,
       view=data_col) %>% 
    # and digest them
    clean_this()
}

# we also save useful intermediate results and
# clean the memory (mostly for my antique laptop)
save_this <- function(x, folder=""){
  tmp <- tempfile("run", paste0("rda/results/", folder), fileext = ".rda")
  obj <- tmp %>% str_remove(".rda") %>% str_remove("^.*/")
  assign(obj, x)
  save(list=obj, file=tmp)
  rm(obj)
  gc()
  beepr::beep(2)
}

# here we wrap all precedent function to run
# on several sample sizes
run_all <- function(db, folder, data_cols, seed_i, epochs=50){
  # available number
  av <- function(x) x$status %>% table %>% min() %>% `-`(100)
  # range of N_model
  N_model_range <- c(50, 100, 150, 200, 300, 400, 500, 600, 700)
  N_model_i <- N_model_range[N_model_range < av(db)]
  
  walk(N_model_i, 
       ~vgg_and_eft_this(db, df_name=folder, data_col=data_cols[1], 
                            N_model=.x, epochs = epochs, seed=seed_i) %>% save_this(folder))
  walk(N_model_i, 
       ~vgg_and_eft_this(db, df_name=folder, data_col=data_cols[2],
                            N_model=.x, epochs = epochs, seed=seed_i) %>% save_this(folder))
}

# this allows to turns a freshly loaded .rda as a list
rda2list <- function(file) {
  e <- new.env()
  load(file, envir = e)
  as.list(e)
}

# end of script #-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#

