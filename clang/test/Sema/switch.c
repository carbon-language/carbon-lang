// RUN: %clang_cc1 -fsyntax-only -verify %s
void f (int z) { 
  while (z) { 
    default: z--;            // expected-error {{statement not in switch}}
  } 
}

void foo(int X) {
  switch (X) {
  case 42: ;                 // expected-note {{previous case}}
  case 5000000000LL:         // expected-warning {{overflow}}
  case 42:                   // expected-error {{duplicate case value}}
   ;

  case 100 ... 99: ;         // expected-warning {{empty case range}}

  case 43: ;                 // expected-note {{previous case}}
  case 43 ... 45:  ;         // expected-error {{duplicate case value}}

  case 100 ... 20000:;       // expected-note {{previous case}}
  case 15000 ... 40000000:;  // expected-error {{duplicate case value}}
  }
}

void test3(void) { 
  // empty switch;
  switch (0); 
}

extern int g();

void test4()
{
  switch (1) {
  case 0 && g():
  case 1 || g():
    break;
  }

  switch(1)  {
  case g(): // expected-error {{expression is not an integer constant expression}}
  case 0 ... g(): // expected-error {{expression is not an integer constant expression}}
    break;
  }
  
  switch (1) {
  case 0 && g() ... 1 || g():
    break;
  }
  
  switch (1) {
  case g() && 0: // expected-error {{expression is not an integer constant expression}} // expected-note {{subexpression not valid in an integer constant expression}}
    break;
  }
  
  switch (1) {
  case 0 ... g() || 1: // expected-error {{expression is not an integer constant expression}} // expected-note {{subexpression not valid in an integer constant expression}}
    break;
  }
}

void test5(int z) { 
  switch(z) {
    default:  // expected-note {{previous case defined here}}
    default:  // expected-error {{multiple default labels in one switch}}
      break;
  }
} 

void test6() {
  const char ch = 'a';
  switch(ch) {
    case 1234:  // expected-warning {{overflow converting case value}}
      break;
  }
}

// PR5606
int f0(int var) { // expected-note{{'var' declared here}}
  switch (va) { // expected-error{{use of undeclared identifier 'va'}}
  case 1:
    break;
  case 2:
    return 1;
  }
  return 2;
}
