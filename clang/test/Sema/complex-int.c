// RUN: clang %s -verify -fsyntax-only

void a() {
__complex__ int arr;
__complex__ short brr;
__complex__ unsigned xx;
__complex__ signed yy;
__complex__ int result;
int ii;

result = arr*ii;
result = ii*brr;

result = arr*brr;
result = xx*yy;

switch (arr) { // expected-error{{statement requires expression of integer type ('_Complex int' invalid)}}
  case brr: ; // expected-error{{case label does not reduce to an integer constant}}
  case xx: ; // expected-error{{case label does not reduce to an integer constant}}
}
}

