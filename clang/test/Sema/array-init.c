// RUN: clang -fsyntax-only -verify -pedantic %s

extern int foof() = 1; // expected-error{{illegal initializer (only variables can be initialized)}}

static int x, y, z;

static int ary[] = { x, y, z }; // expected-error{{initializer element is not a compile-time constant}}
int ary2[] = { x, y, z }; // expected-error{{initializer element is not a compile-time constant}}

extern int fileScopeExtern[3] = { 1, 3, 5 }; // expected-warning{{'extern' variable has an initializer}}

static int ary3[] = { 1, "abc", 3, 4 }; // expected-warning{{incompatible pointer to integer conversion initializing 'char [4]', expected 'int'}}

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

  struct threeElements *p = 7; // expected-warning{{incompatible integer to pointer conversion initializing 'int', expected 'struct threeElements *'}}
  
  extern int blockScopeExtern[3] = { 1, 3, 5 }; // expected-error{{'extern' variable cannot have an initializer}}
  
  static int x2[3] = { 1.0, "abc" , 5.8 }; // expected-warning{{incompatible pointer to integer conversion initializing 'char [4]', expected 'int'}}
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
    { 4, 6, 8 },
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
  int a[][] = { 1, 2 }; // expected-error{{array has incomplete element type 'int []'}}
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
  
  int i[] = { "Hello "}; // expected-warning{{incompatible pointer to integer conversion initializing 'char [7]', expected 'int'}}
  char c2[] = { "Hello", "Good bye" }; //expected-error{{excess elements in char array initializer}}

  int i2[1] = { "Hello" }; //expected-warning{{incompatible pointer to integer conversion initializing 'char [6]', expected 'int'}}
  char c3[5] = { "Hello" };
  char c4[4] = { "Hello" }; //expected-warning{{initializer-string for char array is too long}}

  int i3[] = {}; //expected-warning{{zero size arrays are an extension}} expected-warning{{use of GNU empty initializer extension}}
}

void variableArrayInit() {
  int a = 4;
  char strlit[a] = "foo"; //expected-error{{variable-sized object may not be initialized}}
  int b[a] = { 1, 2, 4 }; //expected-error{{variable-sized object may not be initialized}}
}

// Pure array tests
float r1[10] = {{7}}; //expected-warning{{braces around scalar initializer}}
float r2[] = {{8}}; //expected-warning{{braces around scalar initializer}}
char r3[][5] = {1,2,3,4,5,6};
char r3_2[sizeof r3 == 10 ? 1 : -1];
float r4[1][2] = {1,{2},3,4}; //expected-warning{{braces around scalar initializer}} expected-warning{{excess elements in array initializer}}
char r5[][5] = {"aa", "bbb", "ccccc"};
char r6[sizeof r5 == 15 ? 1 : -1];
const char r7[] = "zxcv";
char r8[5] = "5char";
char r9[5] = "6chars"; //expected-warning{{initializer-string for char array is too long}}

int r11[0] = {}; //expected-warning{{zero size arrays are an extension}} expected-warning{{use of GNU empty initializer extension}}

// Some struct tests
void autoStructTest() {
struct s1 {char a; char b;} t1;
struct s2 {struct s1 c;} t2 = { t1 };
// The following is a less than great diagnostic (though it's on par with EDG).
struct s1 t3[] = {t1, t1, "abc", 0}; //expected-warning{{incompatible pointer to integer conversion initializing 'char [4]', expected 'char'}}
int t4[sizeof t3 == 6 ? 1 : -1];
}
struct foo { int z; } w;
int bar (void) { 
  struct foo z = { w }; //expected-error{{incompatible type initializing 'struct foo', expected 'int'}}
  return z.z; 
} 
struct s3 {void (*a)(void);} t5 = {autoStructTest};
// GCC extension; flexible array init. Once this is implemented, the warning should be removed.
// Note that clang objc implementation depends on this extension.
struct {int a; int b[];} t6 = {1, {1, 2, 3}}; //expected-warning{{excess elements in array initializer}}
union {char a; int b;} t7[] = {1, 2, 3};
int t8[sizeof t7 == (3*sizeof(int)) ? 1 : -1];

struct bittest{int : 31, a, :21, :12, b;};
struct bittest bittestvar = {1, 2, 3, 4}; //expected-warning{{excess elements in array initializer}}

// Not completely sure what should happen here...
int u1 = {}; //expected-warning{{use of GNU empty initializer extension}} expected-error{{scalar initializer cannot be empty}}
int u2 = {{3}}; //expected-error{{too many braces around scalar initializer}}

// PR2362
void varArray() {
  int c[][x] = { 0 }; //expected-error{{variable-sized object may not be initialized}}
}

// PR2151
int emptyInit() {struct {} x[] = {6};} //expected-warning{{empty struct extension}} expected-error{{initializer for aggregate with no elements}}

int noNamedInit() {
struct {int:5;} x[] = {6}; //expected-error{{initializer for aggregate with no elements}}
}
struct {int a; int:5;} noNamedImplicit[] = {1,2,3};
int noNamedImplicitCheck[sizeof(noNamedImplicit) == 3 * sizeof(*noNamedImplicit) ? 1 : -1];


// ptrs are constant
struct soft_segment_descriptor {
	int ssd_base;
};
static int dblfault_tss;

union uniao { int ola; } xpto[1];

struct soft_segment_descriptor gdt_segs[] = {
	{(int) &dblfault_tss},
	{ (int)xpto},
};
