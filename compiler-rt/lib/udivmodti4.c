/* ===-- udivmodti4.c - Implement __udivmodti4 -----------------------------===
 *
 *                    The LLVM Compiler Infrastructure
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *
 * ===----------------------------------------------------------------------===
 *
 * This file implements __udivmodti4 for the compiler_rt library.
 *
 * ===----------------------------------------------------------------------===
 */ 

#if __x86_64

#include "int_lib.h"

/* Effects: if rem != 0, *rem = a % b 
 * Returns: a / b 
 */

/* Translated from Figure 3-40 of The PowerPC Compiler Writer's Guide */

tu_int
__udivmodti4(tu_int a, tu_int b, tu_int* rem)
{
    const unsigned n_udword_bits = sizeof(du_int) * CHAR_BIT;
    const unsigned n_utword_bits = sizeof(tu_int) * CHAR_BIT;
    utwords n;
    n.all = a;
    utwords d;
    d.all = b;
    utwords q;
    utwords r;
    unsigned sr;
    /* special cases, X is unknown, K != 0 */
    if (n.high == 0)
    {
        if (d.high == 0)
        {
            /* 0 X
             * ---
             * 0 X
             */
            if (rem)
                *rem = n.low % d.low;
            return n.low / d.low;
        }
        /* 0 X
         * ---
         * K X
         */
        if (rem)
            *rem = n.low;
        return 0;
    }
    /* n.high != 0 */
    if (d.low == 0)
    {
        if (d.high == 0)
        {
            /* K X
             * ---
             * 0 0
             */
            if (rem)
                *rem = n.high % d.low;
            return n.high / d.low;
        }
        /* d.high != 0 */
        if (n.low == 0)
        {
            /* K 0
             * ---
             * K 0
             */
            if (rem)
            {
                r.high = n.high % d.high;
                r.low = 0;
                *rem = r.all;
            }
            return n.high / d.high;
        }
        /* K K
         * ---
         * K 0
         */
        if ((d.high & (d.high - 1)) == 0)     /* if d is a power of 2 */
        {
            if (rem)
            {
                r.low = n.low;
                r.high = n.high & (d.high - 1);
                *rem = r.all;
            }
            return n.high >> __builtin_ctzll(d.high);
        }
        /* K K
         * ---
         * K 0
         */
        sr = __builtin_clzll(d.high) - __builtin_clzll(n.high);
        /* 0 <= sr <= n_udword_bits - 2 or sr large */
        if (sr > n_udword_bits - 2)
        {
           if (rem)
                *rem = n.all;
            return 0;
        }
        ++sr;
        /* 1 <= sr <= n_udword_bits - 1 */
        /* q.all = n.all << (n_utword_bits - sr); */
        q.low = 0;
        q.high = n.low << (n_udword_bits - sr);
        /* r.all = n.all >> sr; */
        r.high = n.high >> sr;
        r.low = (n.high << (n_udword_bits - sr)) | (n.low >> sr);
    }
    else  /* d.low != 0 */
    {
        if (d.high == 0)
        {
            /* K X
             * ---
             * 0 K
             */
            if ((d.low & (d.low - 1)) == 0)     /* if d is a power of 2 */
            {
                if (rem)
                    *rem = n.low & (d.low - 1);
                if (d.low == 1)
                    return n.all;
                unsigned sr = __builtin_ctzll(d.low);
                q.high = n.high >> sr;
                q.low = (n.high << (n_udword_bits - sr)) | (n.low >> sr);
                return q.all;
            }
            /* K X
             * ---
             * 0 K
             */
            sr = 1 + n_udword_bits + __builtin_clzll(d.low)
                                   - __builtin_clzll(n.high);
            /* 2 <= sr <= n_utword_bits - 1
             * q.all = n.all << (n_utword_bits - sr);
             * r.all = n.all >> sr;
             * if (sr == n_udword_bits)
             * {
             *     q.low = 0;
             *     q.high = n.low;
             *     r.high = 0;
             *     r.low = n.high;
             * }
             * else if (sr < n_udword_bits)  // 2 <= sr <= n_udword_bits - 1
             * {
             *     q.low = 0;
             *     q.high = n.low << (n_udword_bits - sr);
             *     r.high = n.high >> sr;
             *     r.low = (n.high << (n_udword_bits - sr)) | (n.low >> sr);
             * }
             * else              // n_udword_bits + 1 <= sr <= n_utword_bits - 1
             * {
             *     q.low = n.low << (n_utword_bits - sr);
             *     q.high = (n.high << (n_utword_bits - sr)) |
             *              (n.low >> (sr - n_udword_bits));
             *     r.high = 0;
             *     r.low = n.high >> (sr - n_udword_bits);
             * }
             */
            q.low =  (n.low << (n_utword_bits - sr)) &
                     ((di_int)(int)(n_udword_bits - sr) >> (n_udword_bits-1));
            q.high = ((n.low << ( n_udword_bits - sr))                        &
                     ((di_int)(int)(sr - n_udword_bits - 1) >> (n_udword_bits-1))) |
                     (((n.high << (n_utword_bits - sr))                       |
                     (n.low >> (sr - n_udword_bits)))                         &
                     ((di_int)(int)(n_udword_bits - sr) >> (n_udword_bits-1)));
            r.high = (n.high >> sr) &
                     ((di_int)(int)(sr - n_udword_bits) >> (n_udword_bits-1));
            r.low =  ((n.high >> (sr - n_udword_bits))                        &
                     ((di_int)(int)(n_udword_bits - sr - 1) >> (n_udword_bits-1))) |
                     (((n.high << (n_udword_bits - sr))                       |
                     (n.low >> sr))                                           &
                     ((di_int)(int)(sr - n_udword_bits) >> (n_udword_bits-1)));
        }
        else
        {
            /* K X
             * ---
             * K K
             */
            sr = __builtin_clzll(d.high) - __builtin_clzll(n.high);
            /*0 <= sr <= n_udword_bits - 1 or sr large */
            if (sr > n_udword_bits - 1)
            {
               if (rem)
                    *rem = n.all;
                return 0;
            }
            ++sr;
            /* 1 <= sr <= n_udword_bits */
            /* q.all = n.all << (n_utword_bits - sr); */
            q.low = 0;
            q.high = n.low << (n_udword_bits - sr);
            /* r.all = n.all >> sr;
             * if (sr < n_udword_bits)
             * {
             *     r.high = n.high >> sr;
             *     r.low = (n.high << (n_udword_bits - sr)) | (n.low >> sr);
             * }
             * else
             * {
             *     r.high = 0;
             *     r.low = n.high;
             * }
             */
            r.high = (n.high >> sr) &
                     ((di_int)(int)(sr - n_udword_bits) >> (n_udword_bits-1));
            r.low = (n.high << (n_udword_bits - sr)) |
                    ((n.low >> sr)                   &
                    ((di_int)(int)(sr - n_udword_bits) >> (n_udword_bits-1)));
        }
    }
    /* Not a special case
     * q and r are initialized with:
     * q.all = n.all << (n_utword_bits - sr);
     * r.all = n.all >> sr;
     * 1 <= sr <= n_utword_bits - 1
     */
    su_int carry = 0;
    for (; sr > 0; --sr)
    {
        /* r:q = ((r:q)  << 1) | carry */
        r.high = (r.high << 1) | (r.low  >> (n_udword_bits - 1));
        r.low  = (r.low  << 1) | (q.high >> (n_udword_bits - 1));
        q.high = (q.high << 1) | (q.low  >> (n_udword_bits - 1));
        q.low  = (q.low  << 1) | carry;
        /* carry = 0;
         * if (r.all >= d.all)
         * {
         *     r.all -= d.all;
         *      carry = 1;
         * }
         */
        const ti_int s = (ti_int)(d.all - r.all - 1) >> (n_utword_bits - 1);
        carry = s & 1;
        r.all -= d.all & s;
    }
    q.all = (q.all << 1) | carry;
    if (rem)
        *rem = r.all;
    return q.all;
}

#endif
