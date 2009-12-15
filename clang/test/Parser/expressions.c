// RUN: %clang_cc1 -parse-noop -verify %s

void test1() {
  if (sizeof (int){ 1});   // sizeof compound literal
  if (sizeof (int));       // sizeof type

  (int)4;   // cast.
  (int){4}; // compound literal.

  // FIXME: change this to the struct version when we can.
  //int A = (struct{ int a;}){ 1}.a;
  int A = (int){ 1}.a;
}

int test2(int a, int b) {
  return a ? a,b : a;
}

int test3(int a, int b, int c) {
  return a = b = c;
}

int test4() {
  test4();
}

int test_offsetof() {
  // FIXME: change into something that is semantically correct.
  __builtin_offsetof(int, a.b.c[4][5]);
}

void test_sizeof(){
        int arr[10];
        sizeof arr[0];
        sizeof(arr[0]);
        sizeof(arr)[0];
}

// PR3418
int test_leading_extension() {
  __extension__ (*(char*)0) = 1;
}

// PR3972
int test5(int);
int test6(void) { 
  return test5(      // expected-note {{to match}}
               test5(1)
                 ; // expected-error {{expected ')'}}
}
