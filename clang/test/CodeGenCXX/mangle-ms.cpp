// RUN: %clang_cc1 -fms-extensions -fblocks -emit-llvm %s -o - -cxx-abi microsoft -triple=i386-pc-win32 | FileCheck %s

// CHECK: @"\01?a@@3HA"
// CHECK: @"\01?b@N@@3HA"
// CHECK: @c
// CHECK: @"\01?d@foo@@0FB"
// CHECK: @"\01?e@foo@@1JC"
// CHECK: @"\01?f@foo@@2DD"
// CHECK: @"\01?g@bar@@2HA"
// CHECK: @"\01?h@@3QAHA"
// CHECK: @"\01?i@@3PAY0BE@HA"
// CHECK: @"\01?j@@3P6GHCE@ZA"
// CHECK: @"\01?k@@3PTfoo@@DA"
// CHECK: @"\01?l@@3P8foo@@AEHH@ZA"
// CHECK: @"\01?color1@@3PANA"
// CHECK: @"\01?color2@@3PBNA"
// CHECK: @"\01?color3@@3PBY02NA"

int a;

namespace N { int b; }

static int c;
int _c(void) {return c;}
// CHECK: @"\01?_c@@YAHXZ"

class foo {
  static const short d;
protected:
  static volatile long e;
public:
  static const volatile char f;
  int operator+(int a);
  foo(){}
//CHECK: @"\01??0foo@@QAE@XZ"

  ~foo(){}
//CHECK: @"\01??1foo@@QAE@XZ"

  foo(int i){}
//CHECK: @"\01??0foo@@QAE@H@Z"

  foo(char *q){}
//CHECK: @"\01??0foo@@QAE@PAD@Z"
}f,s1(1),s2((char*)0);

typedef foo (foo2);

struct bar {
  static int g;
};

union baz {
  int a;
  char b;
  double c;
};

enum quux {
  qone,
  qtwo,
  qthree
};

int foo::operator+(int a) {return a;}
// CHECK: @"\01??Hfoo@@QAEHH@Z"

const short foo::d = 0;
volatile long foo::e;
const volatile char foo::f = 'C';

int bar::g;

extern int * const h = &a;

int i[10][20];

int (__stdcall *j)(signed char, unsigned char);

const volatile char foo2::*k;

int (foo2::*l)(int);

// Static functions are mangled, too.
// Also make sure calling conventions, arglists, and throw specs work.
static void __stdcall alpha(float a, double b) throw() {}
bool __fastcall beta(long long a, wchar_t b) throw(signed char, unsigned char) {
// CHECK: @"\01?beta@@YI_N_J_W@Z"
  alpha(0.f, 0.0);
  return false;
}

// CHECK: @"\01?alpha@@YGXMN@Z"

// Make sure tag-type mangling works.
void gamma(class foo, struct bar, union baz, enum quux) {}
// CHECK: @"\01?gamma@@YAXVfoo@@Ubar@@Tbaz@@W4quux@@@Z"

// Make sure pointer/reference-type mangling works.
void delta(int * const a, const long &) {}
// CHECK: @"\01?delta@@YAXQAHABJ@Z"

// Array mangling.
void epsilon(int a[][10][20]) {}
// CHECK: @"\01?epsilon@@YAXQAY19BE@H@Z"

// Blocks mangling (Clang extension).
void zeta(int (^)(int, int)) {}
// CHECK: @"\01?zeta@@YAXP_EAHHH@Z@Z"

void operator_new_delete() {
  char *ptr = new char;
// CHECK: @"\01??2@YAPAXI@Z"

  delete ptr;
// CHECK: @"\01??3@YAXPAX@Z"

  char *array = new char[42];
// CHECK: @"\01??_U@YAPAXI@Z"

  delete [] array;
// CHECK: @"\01??_V@YAXPAX@Z"
}

// PR13022
void (redundant_parens)();
void redundant_parens_use() { redundant_parens(); }
// CHECK: @"\01?redundant_parens@@YAXXZ"

// PR13047
typedef double RGB[3];
RGB color1;
extern const RGB color2 = {};
extern RGB const ((color3)[5]) = {};
