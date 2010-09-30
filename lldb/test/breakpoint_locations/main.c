#include <stdio.h>

#define INLINE_ME __inline__ __attribute__((always_inline))

int
func_not_inlined (void)
{
    printf ("Called func_not_inlined.\n");
    return 0;
}

INLINE_ME int
func_inlined (void)
{
    static int func_inline_call_count = 0;
    printf ("Called func_inlined.\n");
    ++func_inline_call_count;
    printf ("Returning func_inlined call count: %d.\n", func_inline_call_count);
    return func_inline_call_count;
}

int
main (int argc, char **argv)
{
  printf ("Starting...\n");

  int (*func_ptr) (void);
  func_ptr = func_inlined;

  int a = func_inlined();
  printf("First call to func_inlined() returns: %d.\n", a);

  func_not_inlined ();

  func_ptr ();

  printf("Last call to func_inlined() returns: %d.\n", func_inlined ());
  return 0;
}


