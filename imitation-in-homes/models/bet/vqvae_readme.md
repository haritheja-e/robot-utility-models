During VQ-VAE pretraining, the hyperparameters to be determined (in order of importance, with the most important at the top):

1. vqvae_n_embed: (10~16) This represents the total possible number of modes, calculated as `vqvae_n_embed^vqvae_groups`. It is the best way to experiment with various values such as 10, 12, and 16 and check for the best performance on the actual robot afterward. But to have more than 80% usage of the total possible mode combinations (`vqvae_n_embed^vqvae_groups`) during pretraining if favorable. In other words, having over 20% of dead combinations is not favorable.

2. action_window_size:

    - 1 (single-step prediction): Generally sufficient for most environments.

    - 3~5 (multi-step prediction): Can be helpful in environments where action correlation, such as in PushT, is important.

3. encoder_loss_multiplier: Adjust this value when the action scale is not between -1 and 1. For example, if the action scale is -100 to 100, a value of 0.01 could be used. If action data is normalized, the default value can be used without adjustment.


Hyperparameters to be determined during the BeT training (in order of importance, with the most important at the top):

1. obs window_size: 10 ~ 100: While 10 is suitable in most cases, consider increasing it if a longer observation history is deemed beneficial.

2. offset_loss_multiplier: If the action scale is around -1 to 1, the most common value of `offset_loss_multiplier` is 100 (default). Adjust this value if the action scale is not between -1 and 1. For example, if the action scale is -100 to 100, a value of 1 could be used.

3. secondary_code_multiplier: The default value is 0.5. Experimenting with values between 0.5 and 3 is recommended. A larger value emphasizes predictions for the secondary code more than offset predictions.

4. sequentially_select: In my experiments, `sequentially_select=False` consistently showed better performance than `sequentially_select=True`.
