// RUN: clang -fsyntax-only %s -verify
// Test case for: 
//   <rdar://problem/6248119> @finally doesn't introduce a new scope

void f0() {
  int i;
  @try { 
  } @finally {
    int i = 0;
  }
}
