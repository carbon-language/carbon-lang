// RUN: %clang_cc1 -fsyntax-only -verify %s

// rdar: // 8125274
static int a16[];  // expected-warning {{tentative array definition assumed to have one element}}

void f16(void) {
    extern int a16[];
}


// PR10013: Scope of extern declarations extend past enclosing block
extern int PR10013_x;
int PR10013(void) {
  int *PR10013_x = 0;
  {
    extern int PR10013_x;
    extern int PR10013_x; 
  }
  
  return PR10013_x; // expected-warning{{incompatible pointer to integer conversion}}
}

static int test1_a[]; // expected-warning {{tentative array definition assumed to have one element}}
extern int test1_a[];

// rdar://13535367
void test2declarer() { extern int test2_array[100]; }
extern int test2_array[];
int test2v = sizeof(test2_array); // expected-error {{invalid application of 'sizeof' to an incomplete type 'int []'}}

void test3declarer() {
  { extern int test3_array[100]; }
  extern int test3_array[];
  int x = sizeof(test3_array); // expected-error {{invalid application of 'sizeof' to an incomplete type 'int []'}}
}

void test4() {
  extern int test4_array[];
  {
    extern int test4_array[100];
    int x = sizeof(test4_array); // fine
  }
  int x = sizeof(test4_array); // expected-error {{invalid application of 'sizeof' to an incomplete type 'int []'}}
}
