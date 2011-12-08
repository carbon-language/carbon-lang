#include <stdio.h>
#include "foo.h"

int 
main ()
{
  struct foo *my_foo_ptr;
  my_foo_ptr = GetMeAFoo();
  // Set breakpoint 0 here.
  printf ("My sub foo has: %d.\n", GetMeASubFoo(my_foo_ptr)->sub_1);

  return 0;
}
