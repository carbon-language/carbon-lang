// RUN: clang -parse-ast-check -pedantic %s

extern int foof() = 1; // expected-error{{illegal initializer (only variables can be initialized)}}

static int x, y, z;

static int ary[] = { x, y, z }; // expected-error{{initializer element is not constant}}
int ary2[] = { x, y, z }; // expected-error{{initializer element is not constant}}

extern int fileScopeExtern[3] = { 1, 3, 5 }; // expected-warning{{'extern' variable has an initializer}}

static int ary3[] = { 1, "abc", 3, 4 }; // expected-warning{{incompatible types assigning 'char *' to 'int'}}

void func() {
  int x = 1;

  typedef int TInt = 1; // expected-error{{illegal initializer (only variables can be initialized)}}

  int xComputeSize[] = { 1, 3, 5 };

  int x3[x] = { 1, 2 }; // expected-error{{variable-sized object may not be initialized}}

  int x4 = { 1, 2 }; // expected-warning{{braces around scalar initializer}} expected-warning{{excess elements in array initializer}}

  int y[4][3] = { 
    { 1, 3, 5 },
    { 2, 4, 6 },
    { 3, 5, 7 },
  };

  int y2[4][3] = {
    1, 3, 5, 2, 4, 6, 3, 5, 7
  };

  int y3[4][3] = {  
    { 1, 3, 5 },
    { 2, 4, 6 },
    { 3, 5, 7 },
    { 4, 6, 8 },
    { 5 }, // expected-warning{{excess elements in array initializer}}
  };

  struct threeElements {
    int a,b,c;
  } z = { 1 };

  struct threeElements *p = 7; // expected-warning{{incompatible types assigning 'int' to 'struct threeElements *'}}
  
  extern int blockScopeExtern[3] = { 1, 3, 5 }; // expected-error{{'extern' variable cannot have an initializer}}
  
  static int x2[3] = { 1.0, "abc" , 5.8 }; // expected-warning{{incompatible types assigning 'char *' to 'int'}}
}

void test() {
  int y1[3] = { 
    { 1, 2, 3 } // expected-warning{{braces around scalar initializer}} expected-warning{{excess elements in array initializer}}
  };
  int y3[4][3] = {  
    { 1, 3, 5 },
    { 2, 4, 6 },
    { 3, 5, 7 },
    { 4, 6, 8 },
    {  }, // expected-warning{{use of GNU empty initializer extension}} expected-warning{{excess elements in array initializer}}
  };
  int y4[4][3] = {  
    { 1, 3, 5, 2 }, // expected-warning{{excess elements in array initializer}}
    { 4, 6 },
    { 3, 5, 7 },
    { 4, 6, 8 }, // expected-warning{{excess elements in array initializer}}
  };
}

void allLegalAndSynonymous() {
  short q[4][3][2] = {
    { 1 },
    { 2, 3 },
    { 4, 5, 6 }
  };
  short q2[4][3][2] = {
    { 1, 0, 0, 0, 0, 0 },
    { 2, 3, 0, 0, 0, 0 },
    { 4, 5, 6 }
  };
  short q3[4][3][2] = {
    { 
      { 1 },
    },
    { 
      { 2, 3 },
    },
    { 
      { 4, 5 },
      { 6 },
    },
  };
}

void legal() {
  short q[][3][2] = {
    { 1 },
    { 2, 3 },
    { 4, 5, 6 }
  };
}

void illegal() {
  short q2[4][][2] = { // expected-error{{array has incomplete element type 'short [][2]'}}
    { 1, 0, 0, 0, 0, 0 },
    { 2, 3, 0, 0, 0, 0 },
    { 4, 5, 6 }
  };
  short q3[4][3][] = { // expected-error{{array has incomplete element type 'short []'}}
    { 
      { 1 },
    },
    { 
      { 2, 3 },
    },
    { 
      { 4, 5 },
      { 6 },
    },
  };
  // FIXME: the following two errors are redundant
  int a[][] = { 1, 2 }; // expected-error{{array has incomplete element type 'int []'}} expected-error{{variable has incomplete type 'int []'}}
}

typedef int AryT[];

void testTypedef()
{
  AryT a = { 1, 2 }, b = { 3, 4, 5 };
}

