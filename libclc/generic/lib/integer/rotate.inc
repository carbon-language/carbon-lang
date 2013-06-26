/**
 * Not necessarily optimal... but it produces correct results (at least for int)
 * If we're lucky, LLVM will recognize the pattern and produce rotate
 * instructions:
 * http://llvm.1065342.n5.nabble.com/rotate-td47679.html
 * 
 * Eventually, someone should feel free to implement an llvm-specific version
 */

_CLC_OVERLOAD _CLC_DEF GENTYPE rotate(GENTYPE x, GENTYPE n){
    //Try to avoid extra work if someone's spinning the value through multiple
    //full rotations
    n = n % (GENTYPE)GENSIZE;

#ifdef SCALAR
    if (n > 0){
        return (x << n) | (((UGENTYPE)x) >> (GENSIZE - n));
    } else if (n == 0){
        return x;
    } else {
        return ( (((UGENTYPE)x) >> -n) | (x << (GENSIZE + n)) );
    }
#else
    //XXX: There's a lot of __builtin_astype calls to cast everything to
    //     unsigned ... This should be improved so that if GENTYPE==UGENTYPE, no
    //     casts are required.
    
    UGENTYPE x_1 = __builtin_astype(x, UGENTYPE);

    UGENTYPE amt;
    amt = (n < (GENTYPE)0 ? __builtin_astype((GENTYPE)0-n, UGENTYPE) : (UGENTYPE)0);
    x_1 = (x_1 >> amt) | (x_1 << ((UGENTYPE)GENSIZE - amt));

    amt = (n < (GENTYPE)0 ? (UGENTYPE)0 : __builtin_astype(n, UGENTYPE));
    x_1 = (x_1 << amt) | (x_1 >> ((UGENTYPE)GENSIZE - amt));

    return __builtin_astype(x_1, GENTYPE);
#endif
}