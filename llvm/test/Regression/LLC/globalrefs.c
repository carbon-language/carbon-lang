/* globalrefs.c - Test symbolic constant expressions constructed from
 * global addresses and index expressions into global addresses.
 * Do this both with global constants and with inline constant.
 * Instead of printing absolute addresses, print out the differences in
 * memory addresses to get output that matches that of the native compiler.
 */

#include <stdio.h>

#define __STDC_LIMIT_MACROS 1
#include <inttypes.h>

struct test {
  long A;
  struct { unsigned X; unsigned Y; } S;
  struct test* next;
};

struct test  TestArray[10];
struct test  Test1;

/* Create global symbolic constants from the addresses of the above globals */

struct test* TestArrayPtr = &TestArray[3]; 
long*        Aptr         = &Test1.A;
unsigned*    Yptr         = &Test1.S.Y;
struct test** NextPtr     = &Test1.next;

void
printdiff(void* p1, void* p2)
{
  printf(" 0x%lx", (unsigned long) p1 - (unsigned long) p2);
}

int
main(int argc, char** argv)
{
  unsigned long diff1, diff2, diff3, diff4; 

  printf("sizeof(struct Test) = %llu\n\n", sizeof(struct test));

  printdiff(&TestArray[3], TestArray);
  printdiff(&Test1.A, &TestArray[3]);
  printdiff(&Test1.S.Y, &Test1.A);
  printdiff(&Test1.next, &Test1.S.Y);
  printf("\n");

  diff1 = (unsigned long) &TestArray[3] - (unsigned long) TestArray;
  diff2 = (unsigned long) &Test1.A - (unsigned long) &TestArray[3];
  diff3 = (unsigned long) &Test1.S.Y - (unsigned long) &Test1.A;
  diff4 = (unsigned long) &Test1.next - (unsigned long) &Test1.S.Y;

  printf("&TestArray[3] - TestArray = 0x%lx\n", diff1);
  printf("Aptr - &TestArray[3] = 0x%lx\n", diff2);
  printf("Xptr - Aptr          = 0x%lx\n", diff3);
  printf("NextPtr - Xptr       = 0x%lx\n\n", diff4);

  diff1 = (unsigned long) TestArrayPtr - (unsigned long) TestArray;
  diff2 = (unsigned long) Aptr - (unsigned long) TestArrayPtr;
  diff3 = (unsigned long) Yptr - (unsigned long) Aptr;
  diff4 = (unsigned long) NextPtr - (unsigned long) Yptr;

  printf("&TestArray[3] - TestArray = 0x%lx\n", diff1);
  printf("Aptr - &TestArray[3] = 0x%lx\n", diff2);
  printf("Xptr - Aptr          = 0x%lx\n", diff3);
  printf("NextPtr - Xptr       = 0x%lx\n\n", diff4);

  return 0;
}
