// RUN: clang -cc1 -fsyntax-only %s -verify
// Test case for: 
//   <rdar://problem/6248119> @finally doesn't introduce a new scope

void f0() {
  int i;
  @try { 
  } @finally {
    int i = 0;
  }
}

void f1() {
  int i;
  @try { 
    int i =0;
  } @finally {
  }
}

void f2() {
  int i;
  @try { 
  } @catch(id e) {
    int i = 0;
  }
}
