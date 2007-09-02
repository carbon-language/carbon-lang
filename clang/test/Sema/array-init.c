// RUN: clang -parse-ast-check -pedantic %s

void func() {
  int x = 1;

  //int x2[] = { 1, 3, 5 };

  int x3[x] = { 1, 2 }; // gcc-error {{variable-sized object may not be initialized}}

  int x4 = { 1, 2 }; // gcc-warning {{excess elements in array initializer}}

  int y[4][3] = { 
    { 1, 3, 5 },
    { 2, 4, 6 },
    { 3, 5, 7 },
  };

  int y2[4][3] = {
    1, 3, 5, 2, 4, 6, 3, 5, 7
  };

  struct threeElements {
    int a,b,c;
  } z = { 1 };

  struct threeElements *p = 7; // expected-warning{{incompatible types assigning 'int' to 'struct threeElements *'}}
}
