// RUN: clang-cc -analyze -checker-cfref -verify %s &&
// RUN: clang-cc -analyze -checker-cfref -analyzer-store=region -verify %s

struct FPRec {
  void (*my_func)(int * x);  
};

int bar(int x);

int f1_a(struct FPRec* foo) {
  int x;
  (*foo->my_func)(&x);
  return bar(x)+1; // no-warning
}

int f1_b() {
  int x;
  return bar(x)+1;  // expected-warning{{Pass-by-value argument in function call is undefined.}}
}

int f2() {
  
  int x;
  
  if (x+1)  // expected-warning{{Branch}}
    return 1;
    
  return 2;  
}

int f2_b() {
  int x;
  
  return ((x+1)+2+((x))) + 1 ? 1 : 2; // expected-warning{{Branch}}
}

int f3(void) {
  int i;
  int *p = &i;
  if (*p > 0) // expected-warning{{Branch condition evaluates to an uninitialized value}}
    return 0;
  else
    return 1;
}

void f4_aux(float* x);
float f4(void) {
  float x;
  f4_aux(&x);
  return x;  // no-warning
}

struct f5_struct { int x; };
void f5_aux(struct f5_struct* s);
int f5(void) {
  struct f5_struct s;
  f5_aux(&s);
  return s.x; // no-warning
}

int ret_uninit() {
  int i;
  int *p = &i;
  return *p;  // expected-warning{{Uninitialized or undefined value returned to caller.}}
}

// <rdar://problem/6451816>
typedef unsigned char Boolean;
typedef const struct __CFNumber * CFNumberRef;
typedef signed long CFIndex;
typedef CFIndex CFNumberType;
typedef unsigned long UInt32;
typedef UInt32 CFStringEncoding;
typedef const struct __CFString * CFStringRef;
extern Boolean CFNumberGetValue(CFNumberRef number, CFNumberType theType, void *valuePtr);
extern CFStringRef CFStringConvertEncodingToIANACharSetName(CFStringEncoding encoding);

CFStringRef rdar_6451816(CFNumberRef nr) {
  CFStringEncoding encoding;
  // &encoding is casted to void*.  This test case tests whether or not
  // we properly invalidate the value of 'encoding'.
  CFNumberGetValue(nr, 9, &encoding);
  return CFStringConvertEncodingToIANACharSetName(encoding); // no-warning
}

