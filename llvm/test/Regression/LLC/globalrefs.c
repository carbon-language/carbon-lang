/* globalrefs.c - Test constant expressions constructed from global
 * addresses and index expressions into global addresses.
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

struct test* TestArrayPtr = TestArray; 
long*        Aptr         = &Test1.A;
unsigned*    Xptr         = &Test1.S.X;

int
main(int argc, char** argv)
{
  void* a1 = (void*) TestArrayPtr;
  void* a2 = (void*) Aptr;
  void* a3 = (void*) Xptr;

#ifdef WANT_ABSOLUTE_ADDRESSES
  printf("Aptr = 0x%lx, Xptr = 0x%lx, TestArrayPtr = 0x%lx\n",
         Aptr, Xptr, TestArrayPtr);
#endif

  printf("Aptr - TestArrayPtr = 0x%lx, Xptr - Aptr = 0x%lx\n",
         (uint64_t) a2 - (uint64_t) a1, (uint64_t) a3 - (uint64_t) a2);
  return 0;
}
