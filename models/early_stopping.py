# add checkpoints and early stopping
%%time
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

earlystopper = EarlyStopping(patience=8, verbose=1)
checkpointer = ModelCheckpoint(filepath = 'model_unet_4ch.hdf5',
                               verbose=1,
                               save_best_only=True, save_weights_only = True)

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                              patience=5, min_lr=0.000001, verbose=1,  cooldown=1)

# new cell for fitting modeland calling earyl stopper
%%time
history = model.fit_generator(gen_train_fast,
                                        validation_data = gen_test_im, validation_steps=1,
                                              steps_per_epoch=30,
                              epochs=100,
                    callbacks=[earlystopper, checkpointer, reduce_lr])
