**In the context of** adding torch as a dependency to blackboxopt, facing the issue the default distribution of torch via PyPi is GPU enabled and comes with GPU specific dependencies, while the CPU only version needs to be installed from a separate custom index, which slows down poetry update by a factor of ~3, **we decided for** using only one index PyPi with the GPU enabled torch **to achieve** faster poetry update, **accepting** bigger environment sizes due to the additional GPU specific dependencies.