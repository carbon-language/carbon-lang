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
  switch (0); // expected-warning {{no case matching constant switch condition '0'}}
}

extern int g();

void test4()
{
  int cond;
  switch (cond) {
  case 0 && g():
  case 1 || g():
    break;
  }

  switch(cond)  {
  case g(): // expected-error {{expression is not an integer constant expression}}
  case 0 ... g(): // expected-error {{expression is not an integer constant expression}}
    break;
  }
  
  switch (cond) {
  case 0 && g() ... 1 || g():
    break;
  }
  
  switch (cond) {
  case g() && 0: // expected-error {{expression is not an integer constant expression}} // expected-note {{subexpression not valid in an integer constant expression}}
    break;
  }
  
  switch (cond) {
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
  char ch = 'a';
  switch(ch) {
    case 1234:  // expected-warning {{overflow converting case value}}
      break;
  }
}

// PR5606
int f0(int var) {
  switch (va) { // expected-error{{use of undeclared identifier 'va'}}
  case 1:
    break;
  case 2:
    return 1;
  }
  return 2;
}

void test7() {
  enum {
    A = 1,
    B
  } a;
  switch(a) { //expected-warning{{enumeration value 'B' not handled in switch}}
    case A:
      break;
  }
  switch(a) {
    case B:
    case A:
      break;
  }
  switch(a) {
    case A:
    case B:
    case 3: // expected-warning{{case value not in enumerated type ''}}
      break;
  }
  switch(a) {
    case A:
    case B:
    case 3 ... //expected-warning{{case value not in enumerated type ''}}
        4: //expected-warning{{case value not in enumerated type ''}}
      break;
  }
  switch(a) {
    case 1 ... 2:
      break;
  }
  switch(a) {
    case 0 ... 2: //expected-warning{{case value not in enumerated type ''}}
      break;
  }
  switch(a) {
    case 1 ... 3: //expected-warning{{case value not in enumerated type ''}}
      break;
  }
  switch(a) {
    case 0 ...  //expected-warning{{case value not in enumerated type ''}}
      3:  //expected-warning{{case value not in enumerated type ''}}
      break;
  }

}

void test8() {
  enum {
    A,
    B,
    C = 1
  } a;
  switch(a) {
    case A:
    case B:
     break;
  }
  switch(a) {
    case A:
    case C:
      break;
  }
  switch(a) { //expected-warning{{enumeration value 'B' not handled in switch}}
    case A:
      break;
  }
}

void test9() {
  enum {
    A = 3,
    C = 1
  } a;
  switch(a) {
    case 0: //expected-warning{{case value not in enumerated type ''}}
    case 1:
    case 2: //expected-warning{{case value not in enumerated type ''}}
    case 3:
    case 4: //expected-warning{{case value not in enumerated type ''}}
      break;
  }
}

void test10() {
  enum {
    A = 10,
    C = 2,
    B = 4,
    D = 12
  } a;
  switch(a) {
    case 0 ...  //expected-warning{{case value not in enumerated type ''}}
	    1:  //expected-warning{{case value not in enumerated type ''}}
    case 2 ... 4:
    case 5 ...  //expected-warning{{case value not in enumerated type ''}}	
	      9:  //expected-warning{{case value not in enumerated type ''}}
    case 10 ... 12:
    case 13 ...  //expected-warning{{case value not in enumerated type ''}}
              16: //expected-warning{{case value not in enumerated type ''}}
      break;
  }
}

void test11() {
  enum {
    A = -1,
    B,
    C
  } a;
  switch(a) { //expected-warning{{enumeration value 'A' not handled in switch}}
    case B:
    case C:
      break;
  }

  switch(a) {
    case B:
    case C:
      break;
      
    default:
      break;
  }
}

void test12() {
  enum {
    A = -1,
    B = 4294967286
  } a;
  switch(a) {
    case A:
    case B:
      break;
  }
}

// <rdar://problem/7643909>
typedef enum {
    val1,
    val2,
    val3
} my_type_t;

int test13(my_type_t t) {
  switch(t) { // expected-warning{{enumeration value 'val3' not handled in switch}}
  case val1:
    return 1;
  case val2:
    return 2;
  }
  return -1;
}

// <rdar://problem/7658121>
enum {
  EC0 = 0xFFFF0000,
  EC1 = 0xFFFF0001,
};

int test14(int a) {
  switch(a) {
  case EC0: return 0;
  case EC1: return 1;
  }
  return 0;
}

void f1(unsigned x) {
  switch (x) {
  case -1: break;
  default: break;
  }
}

void test15() {
  int i = 0;
  switch (1) { // expected-warning {{no case matching constant switch condition '1'}}
  case 0: i = 0; break;
  case 2: i++; break;
  }
}

void test16() {
  const char c = '5';
  switch (c) { // expected-warning {{no case matching constant switch condition '53'}}
  case '6': return;
  }
}

// PR7359
void test17(int x) {
  switch (x >= 17) { // expected-warning {{switch condition has boolean value}}
  case 0: return;
  }

  switch ((int) (x <= 17)) {
  case 0: return;
  }
}
