from .cpp_binding import optimize_brlen as cpp_optimize_brlen
from .cpp_binding import compute_llh as cpp_compute_llh
from .cpp_binding import test_func as cpp_test_func
import re

def optimize_brlen(tree_str, msa, is_root=False, iters=32):
    ret = cpp_optimize_brlen(tree_str, msa["labels"], msa["sequences"], is_root, iters)
    return ret

def compute_llh(tree_str, msa, is_root=False):
    ret = cpp_compute_llh(tree_str, msa["labels"], msa["sequences"], is_root)
    return ret

def test():
    return cpp_test_func()

def treestr_to_tuples(treestr):
    tmp = re.sub(r'(\w+):', r'"\1":', treestr)
    tmp = tmp.replace(':', ",")
    tmp = tmp.rstrip(';')
    return eval(tmp)


def utree2rtree_guided(utree, rtree):

    def fstleaf(node):
        leaf = node
        while isinstance(leaf, tuple):
            leaf = leaf[0]
        return leaf

    ua, ub, uc = utree[:2], utree[2:4], utree[4:]
    ra, rb = rtree[:2], rtree[2:]

    ual = fstleaf(ua)
    ubl = fstleaf(ub)
    ucl = fstleaf(uc)
    ral = fstleaf(ra)
    rbl = fstleaf(rb)

    raal = fstleaf(ra[0][:2]) if isinstance(ra[0], tuple) and len(ra[0]) == 4 else None
    rabl = fstleaf(ra[0][2:]) if isinstance(ra[0], tuple) and len(ra[0]) == 4 else None
    rbal = fstleaf(rb[0][:2]) if isinstance(rb[0], tuple) and len(rb[0]) == 4 else None
    rbbl = fstleaf(rb[0][2:]) if isinstance(rb[0], tuple) and len(rb[0]) == 4 else None

    new_rtree = None
    matched = False

    # (ua, ub), uc
    if (ual, ubl, ucl) == (raal, rabl, rbl):
        new_rtree = ((*ua, *ub), uc[1]/2, uc[0], uc[1]/2)
        matched = True

    # ua, (ub, uc)
    if (ual, ubl, ucl) == (ral, rbal, rbbl):
        assert not matched
        new_rtree = (ua[0], ua[1]/2, (*ub, *uc), ua[1]/2)
        matched = True

    # (ua, uc), ub
    if (ual, ucl, ubl) == (raal, rabl, rbl):
        assert not matched
        new_rtree = ((*ua, *uc), ub[1]/2, ub[0], ub[1]/2)
        matched = True

    # ua, (uc, ub)
    if (ual, ucl, ubl) == (ral, rbal, rbbl):
        assert not matched
        new_rtree = (ua[0], ua[1]/2, (*uc, *ub), ua[1]/2)
        matched = True

    # (ub, ua), uc
    if (ubl, ual, ucl) == (raal, rabl, rbl):
        assert not matched
        new_rtree = ((*ub, *ua), uc[1]/2, uc[0], uc[1]/2)
        matched = True

    # ub, (ua, uc)
    if (ubl, ual, ucl) == (ral, rbal, rbbl):
        assert not matched
        new_rtree = (ub[0], ub[1]/2, (*ua, *uc), ub[1]/2)
        matched = True

    # (ub, uc), ua
    if (ubl, ucl, ual) == (raal, rabl, rbl):
        assert not matched
        new_rtree = ((*ub, *uc), ua[1]/2, ua[0], ua[1]/2)
        matched = True

    # ub, (uc, ua)
    if (ubl, ucl, ual) == (ral, rbal, rbbl):
        assert not matched
        new_rtree = (ub[0], ub[1]/2, (*uc, *ua), ub[1]/2)
        matched = True

    # (uc, ua), ub
    if (ucl, ual, ubl) == (raal, rabl, rbl):
        assert not matched
        new_rtree = ((*uc, *ua), ub[1]/2, ub[0], ub[1]/2)
        matched = True

    # uc, (ua, ub)
    if (ucl, ual, ubl) == (ral, rbal, rbbl):
        assert not matched
        new_rtree = (uc[0], uc[1]/2, (*ua, *ub), uc[1]/2)
        matched = True

    # (uc, ub), ua
    if (ucl, ubl, ual) == (raal, rabl, rbl):
        assert not matched
        new_rtree = ((*uc, *ub), ua[1]/2, ua[0], ua[1]/2)
        matched = True

    # uc, (ub, ua)
    if (ucl, ubl, ual) == (ral, rbal, rbbl):
        assert not matched
        new_rtree = (uc[0], uc[1]/2, (*ub, *ua), uc[1]/2)
        matched = True
    
    try:
        assert matched
    except:
        import pdb; pdb.set_trace()

    return new_rtree

