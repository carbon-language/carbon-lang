/* Test resolvable and unresolvable calls through function pointers:
 * -- both should be retained in function graphs until resolved or until main
 * -- former should get resolved in or before main() and never appear in GG
 * -- latter should remain unresolved in main() and copied to GG
 * -- globals in GG pointed to by latter should be marked I, but not other nodes
 */

#include <stdlib.h>

extern void exit_dummy(int*);

static int X, M, Z;

void makeCalls(void(*GpKnown)(int*), void(*GpUnknown)(int*))
{
  if (Z == 0) GpUnknown(&X);            /* pass to exit_dummy: never resolved */
  else GpKnown(&M);                     /* pass to knownF: resolved in main*/
  ++Z;
  printf("&Z = %p\n", &Z);              /* "known external": resolved here */
}

void knownF(int* Y)
{
  if (Y == 0) knownF(Y);                /* direct call to self: resolved here */
}

int main(int argc, char** argv)
{
  void(*GpKnown)(int*) = knownF;
  void(*GpUnknown)(int*) = exit_dummy;
  Z = argc;
  makeCalls(GpKnown, GpUnknown);
  return 0;
}
