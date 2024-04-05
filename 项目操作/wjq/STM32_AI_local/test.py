def _iterative_update(edict, new_k, new_v):
    for _k, _v in edict.items():
        if _k == new_k:
            edict[new_k] = new_v
            return True
    for _, _v in edict.items():
        if isinstance(_v, (dict, EasyDict)):
            _ret = _iterative_update(_v, new_k, new_v)
            if _ret:
                return True
    return False
configs={}
k='run_dir'
v='runs/flowers/mcunet-5fps/6b+6w/sgd_qas'

_iterative_update(configs, k, v)
ret=_iterative_update(configs, k, v)
print(ret)