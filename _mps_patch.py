import sys
if sys.platform == 'darwin':
    try:
        import torch.utils.data as tud
        _Orig = tud.DataLoader
        class _SafeDL(_Orig):
            def __init__(self, *a, **kw):
                kw['num_workers'] = 0
                kw.pop('prefetch_factor', None)
                kw.pop('persistent_workers', None)
                super().__init__(*a, **kw)
        tud.DataLoader = _SafeDL
        import torch
        try:
            torch.set_num_threads(1)
        except RuntimeError:
            pass
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
    except Exception:
        pass
    import multiprocessing as mp
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
