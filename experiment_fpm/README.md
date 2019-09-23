## August 29, 2019 - MELD FPM Comparison - NeurIPS 2019 Workshop
Comparison for learning illumination for FPM with backprop and meld for NeurIPS 2019:

Experiment #1: Not so good, solutions match, but recons do not look very good... stepsize is too small (0.1)

09:54:31_batch_size=5_stepsize=0.010_loss_fn=mse_optim=adam_num_unrolls=100_alpha=0.100_num_meas=6_num_leds=89_meld=False
09:55:05_batch_size=5_stepsize=0.010_loss_fn=mse_optim=adam_num_unrolls=100_alpha=0.100_num_meas=6_num_leds=89_meld=True


Experiment #2: Similar, but with a larger stepsize (0.2), causes divergence in meld/backprop (solution add checkpointing (9))!

10:31:27_batch_size=5_stepsize=0.010_loss_fn=mse_optim=adam_num_unrolls=100_alpha=0.200_num_meas=6_num_leds=89_meld=False
10:46:25_batch_size=5_stepsize=0.010_loss_fn=mse_optim=adam_num_unrolls=100_alpha=0.200_num_meas=6_num_leds=89_meld=True

Need to highlight this as a limitation of the method...

Experiment #3: larger stepsize (0.5), with cktps every nine

11:04:39_batch_size=5_stepsize=0.010_loss_fn=mse_optim=adam_num_unrolls=100_alpha=0.500_num_meas=6_num_leds=89_meld=False
11:05:53_batch_size=5_stepsize=0.010_loss_fn=mse_optim=adam_num_unrolls=100_alpha=0.500_num_meas=6_num_leds=89_meld=True

Experiment #4: More measurements: darkfield (9) and brightfield (1). MELD version has checkpointing every 10 (constant 750Mb memory limit)

python train.py  --verbose True --num_iter 100 --batch_size 5 --test_freq 3 --step_size 0.01 --num_unrolls 100 --alpha 0.5 --tensorboard True --gpu 3 --num_df 9

11:14:25_batch_size=5_stepsize=0.010_loss_fn=mse_optim=adam_num_unrolls=100_alpha=0.500_num_df=9_num_bf=1_num_leds=89_meld=False

python train.py  --verbose True --num_iter 100 --batch_size 5 --test_freq 3 --step_size 0.01 --num_unrolls 100 --alpha 0.5 --tensorboard True --gpu 3 --num_df 9 --meldFlag=True --memlimit 750

11:21:58_batch_size=5_stepsize=0.010_loss_fn=mse_optim=adam_num_unrolls=100_alpha=0.500_num_df=9_num_bf=1_num_leds=89_meld=True

Experiment #5: More measurements: darkfield (9) and brightfield (1). MELD version has checkpointing every 10 (constant 750Mb memory limit)

python train.py  --verbose True --num_iter 200 --batch_size 5 --test_freq 3 --step_size 0.01 --num_unrolls 100 --alpha 0.5 --tensorboard True --gpu 3 --num_df 9
16:04:50_batch_size=5_stepsize=0.010_loss_fn=mse_optim=adam_num_unrolls=100_alpha=0.500_num_df=9_num_bf=1_num_leds=89_meld=Falses

python train.py  --verbose True --num_iter 200 --batch_size 5 --test_freq 3 --step_size 0.01 --num_unrolls 100 --alpha 0.5 --tensorboard True --gpu 3 --num_df 9 --meldFlag=True --memlimit 750 --T 6
16:05:13_batch_size=5_stepsize=0.010_loss_fn=mse_optim=adam_num_unrolls=100_alpha=0.500_num_df=9_num_bf=1_num_leds=89_meld=True

Experiment #6: More measurements: darkfield (7) and brightfield (1). MELD version has checkpointing every 8 (constant 750Mb memory limit)

python train.py  --verbose True --num_iter 200 --batch_size 5 --test_freq 3 --step_size 0.01 --num_unrolls 100 --alpha 0.4 --tensorboard True --gpu 3 --num_df 7
16:37:02_batch_size=5_stepsize=0.010_loss_fn=mse_optim=adam_num_unrolls=100_alpha=0.400_num_df=7_num_bf=1_num_leds=89_meld=False

python train.py  --verbose True --num_iter 200 --batch_size 5 --test_freq 3 --step_size 0.01 --num_unrolls 100 --alpha 0.4 --tensorboard True --gpu 3 --num_df 7 --meldFlag=True --memlimit 750
16:36:52_batch_size=5_stepsize=0.010_loss_fn=mse_optim=adam_num_unrolls=100_alpha=0.400_num_df=7_num_bf=1_num_leds=89_meld=True

Experiment #7: More measurements: darkfield (7) and brightfield (1). MELD version has checkpointing every 8 (constant 750Mb memory limit) and learning rate is reduced by 75%

python train.py  --verbose True --num_iter 200 --batch_size 5 --test_freq 3 --step_size 0.0025 --num_unrolls 100 --alpha 0.4 --tensorboard True --gpu 3 --num_df 7

python train.py  --verbose True --num_iter 200 --batch_size 5 --test_freq 3 --step_size 0.0025 --num_unrolls 100 --alpha 0.4 --tensorboard True --gpu 3 --num_df 7 --meldFlag=True --memlimit 750
