// RUN: %clang_cc1 -verify -fsyntax-only -fblocks -Wshadow %s

int i;          // expected-note 3 {{previous declaration is here}}

void foo() {
  int pass1;
  int i;        // expected-warning {{declaration shadows a variable in the global scope}} \
                // expected-note {{previous declaration is here}}
  {
    int pass2;
    int i;      // expected-warning {{declaration shadows a local variable}} \
                // expected-note {{previous declaration is here}}
    {
      int pass3;
      int i;    // expected-warning {{declaration shadows a local variable}}
    }
  }

  int sin; // okay; 'sin' has not been declared, even though it's a builtin.
}

// <rdar://problem/7677531>
void (^test1)(int) = ^(int i) { // expected-warning {{declaration shadows a variable in the global scope}} \
                                 // expected-note{{previous declaration is here}}
  {
    int i; // expected-warning {{declaration shadows a local variable}} \
           // expected-note{{previous declaration is here}}
    
    (^(int i) { return i; })(i); //expected-warning {{declaration shadows a local variable}}
  }
};


struct test2 {
  int i;
};

void test3(void) {
  struct test4 {
    int i;
  };
}

void test4(int i) { // expected-warning {{declaration shadows a variable in the global scope}}
}
