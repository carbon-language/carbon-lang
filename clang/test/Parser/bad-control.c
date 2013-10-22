/* RUN: %clang_cc1 -fsyntax-only -verify %s
*/
void foo() { 
  break; /* expected-error {{'break' statement not in loop or switch statement}} */
}

void foo2() { 
  continue; /* expected-error {{'continue' statement not in loop statement}} */
}

int pr8880() {
  int first = 1;
  for ( ; ({ if (first) { first = 0; continue; } 0; }); ) /* expected-error {{'continue' statement not in loop statement}} */
    return 0;
  return 1;
}

int pr8880_2 (int a) {
  int first = a;
  while(({ if (first) { first = 0; continue; } 0; })) /* expected-error {{'continue' statement not in loop statement}} */
    return a;
}

int pr8880_3 (int a) {
  int first = a;
  while(({ if (first) { first = 0; break; } 0; })) /* expected-error {{'break' statement not in loop or switch statement}} */
    return a;
}

int pr8880_4 (int a) {
  int first = a;
  do {
    return a;
  } while(({ if (first) { first = 0; continue; } 0; })); /* expected-error {{'continue' statement not in loop statement}} */
}

int pr8880_5 (int a) {
  int first = a;
  do {
    return a;
  } while(({ if (first) { first = 0; break; } 0; })); /* expected-error {{'break' statement not in loop or switch statement}} */
}

int pr8880_6 (int a) {
  int first = a;
  switch(({ if (first) { first = 0; break; } a; })) { /* expected-error {{'break' statement not in loop or switch statement}} */
  case 2: return a;
  default: return 0;
  }
  return 1;
}

void pr8880_7() {
  for (int i = 0 ; i != 10 ; i++ ) {
    for ( ; ; ({ ++i; continue; })) { // expected-error {{'continue' statement not in loop statement}}
    }
  }
}

// Have to allow 'break' in the third part of 'for' specifier to enable compilation of QT 4.8 macro 'foreach'
void pr17649() {
  for (int i = 0 ; i != 10 ; i++ )
    for ( ; ; ({ ++i; break; })) {
    }
}

void pr8880_9(int x, int y) {
  switch(x) {
  case 1:
    while(({if (y) break; y;})) {} // expected-error {{'break' statement not in loop or switch statement}}
  }
}

void pr8880_10(int x, int y) {
  while(x > 0) {
    switch(({if(y) break; y;})) { // expected-error {{'break' statement not in loop or switch statement}}
    case 2: x=0;
    }
  }
}

void pr8880_11() {
  for (int i = 0 ; i != 10 ; i++ ) {
    while(({if (i) break; i;})) {} // expected-error {{'break' statement not in loop or switch statement}}
  }
}

// Moved from Analysis/dead-stores.c
void rdar8014335() {
  for (int i = 0 ; i != 10 ; ({ break; })) {
    for ( ; ; ({ ++i; break; })) ;
    i = i * 3;
  }
}

void pr17649_2() {
  for (int i = 0 ; i != 10 ; ({ continue; })) { // expected-error {{'continue' statement not in loop statement}}
    for ( ; ; ({ ++i; continue; })) ; // expected-error {{'continue' statement not in loop statement}}
    i = i * 3;
  }
}
