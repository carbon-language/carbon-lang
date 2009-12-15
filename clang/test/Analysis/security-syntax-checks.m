// RUN: %clang_cc1 -triple i386-apple-darwin10 -analyze -warn-security-syntactic %s -verify

// <rdar://problem/6336718> rule request: floating point used as loop 
//  condition (FLP30-C, FLP-30-CPP)
//
// For reference: https://www.securecoding.cert.org/confluence/display/seccode/FLP30-C.+Do+not+use+floating+point+variables+as+loop+counters
//
void test_float_condition() {
  for (float x = 0.1f; x <= 1.0f; x += 0.1f) {} // expected-warning{{Variable 'x' with floating point type 'float'}}
  for (float x = 100000001.0f; x <= 100000010.0f; x += 1.0f) {} // expected-warning{{Variable 'x' with floating point type 'float'}}
  for (float x = 100000001.0f; x <= 100000010.0f; x++ ) {} // expected-warning{{Variable 'x' with floating point type 'float'}}
  for (double x = 100000001.0; x <= 100000010.0; x++ ) {} // expected-warning{{Variable 'x' with floating point type 'double'}}
  for (double x = 100000001.0; ((x)) <= 100000010.0; ((x))++ ) {} // expected-warning{{Variable 'x' with floating point type 'double'}}
  
  for (double x = 100000001.0; 100000010.0 >= x; x = x + 1.0 ) {} // expected-warning{{Variable 'x' with floating point type 'double'}}
  
  int i = 0;
  for (double x = 100000001.0; ((x)) <= 100000010.0; ((x))++, ++i ) {} // expected-warning{{Variable 'x' with floating point type 'double'}}
  
  typedef float FooType;
  for (FooType x = 100000001.0f; x <= 100000010.0f; x++ ) {} // expected-warning{{Variable 'x' with floating point type 'FooType'}}
}

// <rdar://problem/6335715> rule request: gets() buffer overflow
// Part of recommendation: 300-BSI (buildsecurityin.us-cert.gov)
char* gets(char *buf);

void test_gets() {
  char buff[1024];
  gets(buff); // expected-warning{{Call to function 'gets' is extremely insecure as it can always result in a buffer overflow}}
}

int getpw(unsigned int uid, char *buf);

void test_getpw() {
  char buff[1024];
  getpw(2, buff); // expected-warning{{The getpw() function is dangerous as it may overflow the provided buffer. It is obsoleted by getpwuid().}}
}

// <rdar://problem/6337132> CWE-273: Failure to Check Whether Privileges Were
//  Dropped Successfully
typedef unsigned int __uint32_t;
typedef __uint32_t __darwin_uid_t;
typedef __uint32_t __darwin_gid_t;
typedef __darwin_uid_t uid_t;
typedef __darwin_gid_t gid_t;
int setuid(uid_t);
int setregid(gid_t, gid_t);
int setreuid(uid_t, uid_t);
extern void check(int);

void test_setuid() 
{
  setuid(2); // expected-warning{{The return value from the call to 'setuid' is not checked.  If an error occurs in 'setuid', the following code may execute with unexpected privileges}}
  setuid(0); // expected-warning{{The return value from the call to 'setuid' is not checked.  If an error occurs in 'setuid', the following code may execute with unexpected privileges}}
  if (setuid (2) != 0)
    abort();

  // Currently the 'setuid' check is not flow-sensitive, and only looks
  // at whether the function was called in a compound statement.  This
  // will lead to false negatives, but there should be no false positives.
  int t = setuid(2);  // no-warning
  (void)setuid (2); // no-warning

  check(setuid (2)); // no-warning

  setreuid(2,2); // expected-warning{{The return value from the call to 'setreuid' is not checked.  If an error occurs in 'setreuid', the following code may execute with unexpected privileges}}
  setregid(2,2); // expected-warning{{The return value from the call to 'setregid' is not checked.  If an error occurs in 'setregid', the following code may execute with unexpected privileges}}
}

// <rdar://problem/6337100> CWE-338: Use of cryptographically weak prng
int      rand(void);
double   drand48(void);
double   erand48(unsigned short[3]);
long     jrand48(unsigned short[3]);
void     lcong48(unsigned short[7]);
long     lrand48(void);
long     mrand48(void);
long     nrand48(unsigned short[3]);
long     random(void);
int      rand_r(unsigned *);

void test_rand()
{
  unsigned short a[7];
  unsigned b;
  
  rand();	// expected-warning{{Function 'rand' is obsolete because it implements a poor random number generator.  Use 'arc4random' instead}}
  drand48();	// expected-warning{{Function 'drand48' is obsolete because it implements a poor random number generator.  Use 'arc4random' instead}}
  erand48(a);	// expected-warning{{Function 'erand48' is obsolete because it implements a poor random number generator.  Use 'arc4random' instead}}
  jrand48(a);	// expected-warning{{Function 'jrand48' is obsolete because it implements a poor random number generator.  Use 'arc4random' instead}}
  lcong48(a);	// expected-warning{{Function 'lcong48' is obsolete because it implements a poor random number generator.  Use 'arc4random' instead}}
  lrand48();	// expected-warning{{Function 'lrand48' is obsolete because it implements a poor random number generator.  Use 'arc4random' instead}}
  mrand48();	// expected-warning{{Function 'mrand48' is obsolete because it implements a poor random number generator.  Use 'arc4random' instead}}
  nrand48(a);	// expected-warning{{Function 'nrand48' is obsolete because it implements a poor random number generator.  Use 'arc4random' instead}}
  rand_r(&b);	// expected-warning{{Function 'rand_r' is obsolete because it implements a poor random number generator.  Use 'arc4random' instead}}
  random();	// expected-warning{{The 'random' function produces a sequence of values that an adversary may be able to predict.  Use 'arc4random' instead}}
}

char *mktemp(char *buf);

void test_mktemp() {
  char *x = mktemp("/tmp/zxcv"); // expected-warning{{Call to function 'mktemp' is insecure as it always creates or uses insecure temporary file}}
}
