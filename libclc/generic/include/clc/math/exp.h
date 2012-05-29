#undef exp

// exp(x) = exp2(x * log2(e)
#define exp(val) (__clc_exp2((val) * 1.44269504f))
