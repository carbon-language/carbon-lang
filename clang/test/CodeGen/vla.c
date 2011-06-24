// RUN: %clang_cc1 -triple i386-unknown-unknown %s -emit-llvm -o - | FileCheck %s

int b(char* x);

// Extremely basic VLA test
void a(int x) {
  char arry[x];
  arry[0] = 10;
  b(arry);
}

int c(int n)
{
  return sizeof(int[n]);
}

int f0(int x) {
  int vla[x];
  return vla[x-1];
}

void
f(int count)
{
 int a[count];

  do {  } while (0);

  if (a[0] != 3) {
  }
}

void g(int count) {
  // Make sure we emit sizes correctly in some obscure cases
  int (*a[5])[count];
  int (*b)[][count];
}

// rdar://8403108
// CHECK: define void @f_8403108
void f_8403108(unsigned x) {
  // CHECK: call i8* @llvm.stacksave()
  char s1[x];
  while (1) {
    // CHECK: call i8* @llvm.stacksave()
    char s2[x];
    if (1)
      break;
  // CHECK: call void @llvm.stackrestore(i8*
  }
  // CHECK: call void @llvm.stackrestore(i8*
}

// pr7827
void function(short width, int data[][width]) {} // expected-note {{passing argument to parameter 'data' here}}

void test() {
     int bork[4][13];
     // CHECK: call void @function(i16 signext 1, i32* null)
     function(1, 0);
     // CHECK: call void @function(i16 signext 1, i32* inttoptr
     function(1, 0xbadbeef); // expected-warning {{incompatible integer to pointer conversion passing}}
     // CHECK: call void @function(i16 signext 1, i32* {{.*}})
     function(1, bork);
}

void function1(short width, int data[][width][width]) {}
void test1() {
     int bork[4][13][15];
     // CHECK: call void @function1(i16 signext 1, i32* {{.*}})
     function1(1, bork);
     // CHECK: call void @function(i16 signext 1, i32* {{.*}}) 
     function(1, bork[2]);
}

// rdar://8476159
static int GLOB;
int test2(int n)
{
  GLOB = 0;
  char b[1][n+3];			/* Variable length array.  */
  // CHECK:  [[tmp_1:%.*]] = load i32* @GLOB, align 4
  // CHECK-NEXT: add nsw i32 [[tmp_1]], 1
  __typeof__(b[GLOB++]) c;
  return GLOB;
}

// http://llvm.org/PR8567
// CHECK: define double @test_PR8567
double test_PR8567(int n, double (*p)[n][5]) {
  // CHECK:      [[NV:%.*]] = alloca i32, align 4
  // CHECK-NEXT: [[PV:%.*]] = alloca [5 x double]*, align 4
  // CHECK-NEXT: store
  // CHECK-NEXT: store
  // CHECK-NEXT: [[N:%.*]] = load i32* [[NV]], align 4
  // CHECK-NEXT: [[P:%.*]] = load [5 x double]** [[PV]], align 4
  // CHECK-NEXT: [[T0:%.*]] = mul i32 1, [[N]]
  // CHECK-NEXT: [[T1:%.*]] = getelementptr inbounds [5 x double]* [[P]], i32 [[T0]]
  // CHECK-NEXT: [[T2:%.*]] = getelementptr inbounds [5 x double]* [[T1]], i32 2
  // CHECK-NEXT: [[T3:%.*]] = getelementptr inbounds [5 x double]* [[T2]], i32 0, i32 3
  // CHECK-NEXT: [[T4:%.*]] = load double* [[T3]]
  // CHECK-NEXT: ret double [[T4]]
 return p[1][2][3];
}
