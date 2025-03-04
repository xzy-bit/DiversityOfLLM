# Triton Implementation of GEM Loss

This folder contains the Triton implementation of GEM loss.

## Test

We have successfully tested the implementation. 

To run the tests, you can use the following command:

```bash
python tests/test_gem_loss_triton.py
python tests/test_gem_loss_triton_distributed.py
```

Please contact Ziniu Li (ziniuli@link.cuhk.edu.cn) if you find any issues.


## To Do

- [ ] Add the implementation of GEM with $h = logsigmoid$ (currently only $h = linear$ is supported).
- [ ] Add more tests.


## Acknowledgement

We thank the authors of [flash-attention](https://github.com/Dao-AILab/flash-attention) for providing the Triton implementation of CE loss, for which the GEM loss is based on.
