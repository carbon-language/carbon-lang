// RUN: clang -fsyntax-only -verify -pedantic %s

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

unsigned char asso_values[] = { 34 };
int legal2() { 
  return asso_values[0]; 
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

static char const xx[] = "test";
static char const yy[5] = "test";
static char const zz[3] = "test"; // expected-warning{{initializer-string for char array is too long}}

void charArrays()
{
	static char const test[] = "test";
	static char const test2[] = { "weird stuff" };
	static char const test3[] = { "test", "excess stuff" }; // expected-error{{excess elements in char array initializer}}

  char* cp[] = { "Hello" };

  char c[] = { "Hello" };
  int l[sizeof(c) == 6 ? 1 : -1];
  
  int i[] = { "Hello "}; // expected-error{{array of wrong type 'int' initialized from string constant}}
  char c2[] = { "Hello", "Good bye" }; //expected-error{{excess elements in char array initializer}}

  int i2[1] = { "Hello" }; //expected-error{{array of wrong type 'int' initialized from string constant}}
  char c3[5] = { "Hello" };
  char c4[4] = { "Hello" }; //expected-warning{{initializer-string for char array is too long}}

  int i3[] = {}; //expected-error{{at least one initializer value required to size array}} expected-warning{{use of GNU empty initializer extension}}
}

