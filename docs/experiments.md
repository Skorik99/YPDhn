# ⚙️ Experiments setup

⚠️ **Important**: Follow the [Prerequisites](../README.md#-prerequisites) steps to set up enviroment.

⚠️ **Script argument**: all `scripts` have a `--device_id`, which defines the GPU idx.

## Byzantines Attacks with CIFAR-10 
Figure 1 and Appendix A.2

```bash
cd scripts/
python cifar10_script.py > cifar10_log_script.txt &
```

## Byzantines Attacks with CIFAR-10 under Dirichlet distribution
Figure 2 and Appendix A.3

```bash
cd scripts/
python dirichlet_experiments.py > dirichlet_log_script.txt &
```

## Byzantines Attacks with CIFAR-10 over 100 clients
Appendix A.3

```bash
cd scripts/
python 100_clients_script.py > 100_clients_log_script.txt &
```

## Byzantines Attacks with CIFAR-10 and various trust size
Appendix A.2

```bash
cd scripts/
python trial_size_script.py > trial_size_log_script.txt &
```

## Byzantines Attacks with CIFAR-10 and various number of iteration steps
Varying iteration steps in AutoBANT auxiliary subproblem in Appendix A.2

```bash
cd scripts/
python delta_error_script.py > delta_error_log_script.txt &
```