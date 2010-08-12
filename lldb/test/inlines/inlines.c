#include <stdio.h>

#define INLINE_ME __inline__ __attribute__((always_inline))

INLINE_ME int
inner_inline (int inner_input, int mod_value)
{
  int inner_result;
  inner_result = inner_input % mod_value;
  printf ("Returning: %d.\n", inner_result);
  return inner_result;
}

INLINE_ME int
outer_inline (int outer_input)
{
  int outer_result;

  outer_result = inner_inline (outer_input, outer_input % 3);
  return outer_result;
}

int
main (int argc, char **argv)
{
  printf ("Starting...\n");

  int (*func_ptr) (int);
  func_ptr = outer_inline;

  outer_inline (argc);

  func_ptr (argc);

  return 0;
}


