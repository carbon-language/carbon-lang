
#include <setjmp.h>

sigjmp_buf B;
int foo() {
  sigsetjmp(B, 1);
  bar();
}
