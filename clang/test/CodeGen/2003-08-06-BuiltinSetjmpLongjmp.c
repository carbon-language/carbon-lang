/* RUN: %clang_cc1  %s -emit-llvm -o - | FileCheck %s
 *
 * __builtin_longjmp/setjmp should get transformed into intrinsics.
 */

// CHECK-NOT: builtin_longjmp

void jumpaway(int *ptr) {
  __builtin_longjmp(ptr,1);
}
    
int main(void) {
  __builtin_setjmp(0);
  jumpaway(0);
}
