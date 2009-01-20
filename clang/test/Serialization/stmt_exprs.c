// RUN: clang %s --test-pickling 2>&1 | grep -q 'SUCCESS'
typedef unsigned __uint32_t;

#define __byte_swap_int_var(x) \
__extension__ ({ register __uint32_t __X = (x); \
   __asm ("bswap %0" : "+r" (__X)); \
   __X; })

int test(int _x) {
 return (__byte_swap_int_var(_x));
}
