// RUN: clang-cc %s -verify -fsyntax-only

void a() {
__complex__ int arr;
__complex__ short brr;
__complex__ unsigned xx;
__complex__ signed yy;
__complex__ int result;
int ii;
int aa = 1 + 1.0iF;

result = arr*ii;
result = ii*brr;

result = arr*brr;
result = xx*yy;

switch (arr) { // expected-error{{statement requires expression of integer type ('_Complex int' invalid)}}
  case brr: ; // expected-error{{expression is not an integer constant expression}}
  case xx: ; // expected-error{{expression is not an integer constant expression}}
}
}

void Tester() {
__complex short a1;
__complex int a2;
__complex float a3;
__complex double a4;
short a5;
int a6;
float a7;
double a8;
#define TestPair(m,n) int x##m##n = a##m+a##n;
#define TestPairs(m) TestPair(m,1) TestPair(m,2) \
                    TestPair(m,3) TestPair(m,4) \
                    TestPair(m,5) TestPair(m,6) \
                    TestPair(m,7) TestPair(m,8)
TestPairs(1); TestPairs(2);
TestPairs(3); TestPairs(4);
TestPairs(5); TestPairs(6);
TestPairs(7); TestPairs(8);
}

// rdar://6097730
void test3(_Complex int *x) {
  *x = ~*x;
}		

void test4(_Complex float *x) {
  *x = ~*x;
}		

