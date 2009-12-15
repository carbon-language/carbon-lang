// RUN: %clang_cc1 -fsyntax-only -verify %s
// rdar://6425550
int bar();
void do_something(int);
int *get_ptr();

int foo() {
  if (int X = bar()) {
    return X;
  } else {
    do_something(X); // expected-warning{{'X' is always zero in this context}}
    return 0;
  }
}

bool foo2() {
  if (bool B = bar()) {
    if (int Y = bar()) {
      return B;
    } else {
      do_something(Y); // expected-warning{{'Y' is always zero in this context}}
      return B;
    }
  } else {
    if (bool B2 = B) { // expected-warning{{'B' is always false in this context}}
      do_something(B); // expected-warning{{'B' is always false in this context}}
    } else if (B2) {  // expected-warning{{'B2' is always false in this context}}
      do_something(B); // expected-warning{{'B' is always false in this context}}
      do_something(B2); // expected-warning{{'B2' is always false in this context}}
    }
    return B; // expected-warning{{'B' is always false in this context}}
  }
}

void foo3() {  
  if (int *P1 = get_ptr())
    do_something(*P1);
  else if (int *P2 = get_ptr()) {
    do_something(*P1); // expected-warning{{'P1' is always NULL in this context}}
    do_something(*P2);
  } else {
    do_something(*P1); // expected-warning{{'P1' is always NULL in this context}}
    do_something(*P2); // expected-warning{{'P2' is always NULL in this context}}
  }
}
