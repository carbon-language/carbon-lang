// XFAIL:*
//// Currently, LowerDbgDeclare doesn't lower dbg.declares pointing at allocas
//// for structs.

// REQUIRES: lldb
// UNSUPPORTED: system-windows
// RUN: %dexter --fail-lt 1.0 -w --debugger lldb \
// RUN:     --builder clang-c --cflags "-O2 -glldb" -- %s
//
//// Check debug-info for the escaped struct variable num is reasonable.

#include <stdio.h>
struct Nums { int a, b, c; };
struct Nums glob;
__attribute__((__noinline__))
void esc(struct Nums* nums) {
  glob = *nums;
}

__attribute__((__noinline__))
int main() {
  struct Nums nums = { .c=1 };       //// Dead store.
  printf("s1 nums.c: %d\n", nums.c); // DexLabel('s1')

  nums.c = 2;                        //// Killing store.
  printf("s2 nums.c: %d\n", nums.c); // DexLabel('s2')

  esc(&nums);                        //// Force nums to live on the stack.
  return 0;                          // DexLabel('s3')
}

// DexExpectWatchValue('nums.c', '1', on_line=ref('s1'))
// DexExpectWatchValue('nums.c', '2', from_line=ref('s2'), to_line=ref('s3'))
