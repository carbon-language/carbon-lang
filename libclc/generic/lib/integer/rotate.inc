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
    
    //Determine if we're doing a right or left shift on each component
    //The actual shift algorithm is based on a rotate right
    //e.g. a rotate of int by 5 bits becomes rotate right by 26 bits
    //     and a rotate of int by -4 bits becomes rotate right by 4
    GENTYPE amt = (n > (GENTYPE)0 ? (GENTYPE)GENSIZE - n : (GENTYPE)0 - n );
    
    //Calculate the bits that will wrap
    GENTYPE mask = ( (GENTYPE)1 << amt ) - (GENTYPE)1;
    GENTYPE wrapped_bits = x & mask;
    
    //Shift the input value right and then AND a mask that eliminates
    //sign-extension interference
    //if the rotate amount is 0, just use ~0 for a mask
    GENTYPE se_mask = !amt ? ~((GENTYPE)0) : 
        ( ( (GENTYPE)1 << ((GENTYPE)GENSIZE - amt) ) - (GENTYPE)1 );
    GENTYPE unwrapped_bits = x >> amt;
    unwrapped_bits &= se_mask;
    
    //Finally shift the input right after moving the wrapped bits into position
    return unwrapped_bits | (wrapped_bits << ( (GENTYPE)GENSIZE - amt ) );
}