// RUN: clang-cc -fsyntax-only -verify %s

int test1(int x) {
  goto L;    // expected-error{{illegal goto into protected scope}}
  int a[x];  // expected-note {{jump bypasses initialization of variable length array}}
  int b[x];  // expected-note {{jump bypasses initialization of variable length array}}
  L:
  return sizeof a;
}

int test2(int x) {
  goto L;            // expected-error{{illegal goto into protected scope}}
  typedef int a[x];  // expected-note {{jump bypasses initialization of VLA typedef}}
  L:
  return sizeof(a);
}

void test3clean(int*);

int test3() {
  goto L;            // expected-error{{illegal goto into protected scope}}
int a __attribute((cleanup(test3clean))); // expected-note {{jump bypasses initialization of declaration with __attribute__((cleanup))}}
L:
  return a;
}

int test4(int x) {
  goto L;       // expected-error{{illegal goto into protected scope}}
int a[x];       // expected-note {{jump bypasses initialization of variable length array}}
  test4(x);
L:
  return sizeof a;
}

int test5(int x) {
  int a[x];
  test5(x);
  goto L;  // Ok.
L:
  goto L;  // Ok.
  return sizeof a;
}

int test6() { 
  // just plain invalid.
  goto x;  // expected-error {{use of undeclared label 'x'}}
}

void test7(int x) {
  switch (x) {
  case 1: ;
    int a[x];       // expected-note {{jump bypasses initialization of variable length array}}
  case 2:           // expected-error {{illegal switch case into protected scope}}
    a[1] = 2;
    break;
  }
}

int test8(int x) {
  // For statement.
  goto L2;     // expected-error {{illegal goto into protected scope}}
  for (int arr[x];   // expected-note {{jump bypasses initialization of variable length array}}  
       ; ++x)
  L2:;

  // Statement expressions.
  goto L3;   // expected-error {{illegal goto into protected scope}}
  int Y = ({  int a[x];   // expected-note {{jump bypasses initialization of variable length array}}  
           L3: 4; });
  
  goto L4; // expected-error {{illegal goto into protected scope}}
  {
    int A[x],  // expected-note {{jump bypasses initialization of variable length array}}
        B[x];  // expected-note {{jump bypasses initialization of variable length array}}
  L4: ;
  }
  
  {
  L5: ;// ok
    int A[x], B = ({ if (x)
                       goto L5;
                     else 
                       goto L6;
                   4; }); 
  L6:; // ok.
  }
  
  {
  L7: ;// ok
    int A[x], B = ({ if (x)
                       goto L7;
                     else 
                       goto L8;  // expected-error {{illegal goto into protected scope}}
                     4; }),
        C[x];   // expected-note {{jump bypasses initialization of variable length array}}
  L8:; // bad
  }
 
  
  // Statement expressions 2.
  goto L1;     // expected-error {{illegal goto into protected scope}}
  return x == ({
                 int a[x];   // expected-note {{jump bypasses initialization of variable length array}}  
               L1:
                 42; });
}
