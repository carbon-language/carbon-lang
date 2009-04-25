// RUN: clang-cc %s -verify -pedantic -fsyntax-only

typedef short short_fixed;

enum
{
        // 8.8 short_fixed
        SHORT_FIXED_FRACTIONAL_BITS= 8,
        SHORT_FIXED_ONE= 1<<SHORT_FIXED_FRACTIONAL_BITS
};

#define FLOAT_TO_SHORT_FIXED(f) ((short_fixed)((f)*SHORT_FIXED_ONE))

enum
{
        SOME_VALUE= FLOAT_TO_SHORT_FIXED(0.1) // expected-warning{{expression is not integer constant expression}}
};
