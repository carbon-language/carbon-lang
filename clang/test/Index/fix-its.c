// RUN: c-index-test -test-load-source all -fspell-checking %s 2> %t  
// RUN: FileCheck %s < %t
struct X {
  int wibble;
};

#define MACRO(X) X

void f(struct X *x) {
  // CHECK: error: no member named 'wobble' in 'struct X'; did you mean 'wibble'?
  // CHECK-NOT: FIX-IT
  // CHECK: note: 'wibble' declared here
  MACRO(x->wobble = 17);
  // CHECK: error: no member named 'wabble' in 'struct X'; did you mean 'wibble'?
  // CHECK: FIX-IT: Replace [17:6 - 17:12] with "wibble"
  // CHECK: note: 'wibble' declared here
  x->wabble = 17;
}
