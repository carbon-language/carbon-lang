// RUN: %clang_cc1 -fsyntax-only -verify %s
// RUN: %clang_cc1 -fsyntax-only -x c++ -Werror %s

int pr8880_1() {
  int first = 1;
  for ( ; ({ if (first) { first = 0; continue; } 0; }); )
    return 0;
  return 1;
}

void pr8880_2(int first) {
  for ( ; ({ if (first) { first = 0; break; } 0; }); ) {}
}

void pr8880_3(int first) {
  for ( ; ; (void)({ if (first) { first = 0; continue; } 0; })) {}
}

void pr8880_4(int first) {
  for ( ; ; (void)({ if (first) { first = 0; break; } 0; })) {}
}

void pr8880_5 (int first) {
  while(({ if (first) { first = 0; continue; } 0; })) {}
}

void pr8880_6 (int first) {
  while(({ if (first) { first = 0; break; } 0; })) {}
}

void pr8880_7 (int first) {
  do {} while(({ if (first) { first = 0; continue; } 0; }));
}

void pr8880_8 (int first) {
  do {} while(({ if (first) { first = 0; break; } 0; }));
}

void pr8880_10(int i) {
  for ( ; i != 10 ; i++ )
    for ( ; ; (void)({ ++i; continue; i;})) {} // expected-warning{{'continue' is bound to current loop, GCC binds it to the enclosing loop}}
}

void pr8880_11(int i) {
  for ( ; i != 10 ; i++ )
    for ( ; ; (void)({ ++i; break; i;})) {} // expected-warning{{'break' is bound to current loop, GCC binds it to the enclosing loop}}
}

void pr8880_12(int i, int j) {
  for ( ; i != 10 ; i++ )
    for ( ; ({if (i) continue; i;}); j++) {} // expected-warning {{'continue' is bound to current loop, GCC binds it to the enclosing loop}}
}

void pr8880_13(int i, int j) {
  for ( ; i != 10 ; i++ )
    for ( ; ({if (i) break; i;}); j++) {} // expected-warning{{'break' is bound to current loop, GCC binds it to the enclosing loop}}
}

void pr8880_14(int i) {
  for ( ; i != 10 ; i++ )
    while(({if (i) break; i;})) {} // expected-warning {{'break' is bound to current loop, GCC binds it to the enclosing loop}}
}

void pr8880_15(int i) {
  while (--i)
    while(({if (i) continue; i;})) {} // expected-warning {{'continue' is bound to current loop, GCC binds it to the enclosing loop}}
}

void pr8880_16(int i) {
  for ( ; i != 10 ; i++ )
    do {} while(({if (i) break; i;})); // expected-warning {{'break' is bound to current loop, GCC binds it to the enclosing loop}}
}

void pr8880_17(int i) {
  for ( ; i != 10 ; i++ )
    do {} while(({if (i) continue; i;})); // expected-warning {{'continue' is bound to current loop, GCC binds it to the enclosing loop}}
}

void pr8880_18(int x, int y) {
  while(x > 0)
    switch(({if(y) break; y;})) {
    case 2: x = 0;
    }
}

void pr8880_19(int x, int y) {
  switch(x) {
  case 1:
    switch(({if(y) break; y;})) {
    case 2: x = 0;
    }
  }
}

void pr8880_20(int x, int y) {
  switch(x) {
  case 1:
    while(({if (y) break; y;})) {} //expected-warning {{'break' is bound to loop, GCC binds it to switch}}
  }
}

void pr8880_21(int x, int y) {
  switch(x) {
  case 1:
    do {} while(({if (y) break; y;})); //expected-warning {{'break' is bound to loop, GCC binds it to switch}}
  }
}

void pr8880_22(int x, int y) {
  switch(x) {
  case 1:
    for ( ; ; (void)({ ++y; break; y;})) {} // expected-warning{{'break' is bound to loop, GCC binds it to switc}}
  }
}

void pr8880_23(int x, int y) {
  switch(x) {
  case 1:
    for ( ; ({ ++y; break; y;}); ++y) {} // expected-warning{{'break' is bound to loop, GCC binds it to switch}}
  }
}
