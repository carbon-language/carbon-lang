// RUN: clang-cc -analyze -checker-cfref -analyzer-store=basic -analyzer-constraints=basic -verify %s
// RUN: clang-cc -analyze -checker-cfref -analyzer-store=basic -analyzer-constraints=range -verify %s
// RUN: clang-cc -analyze -checker-cfref -analyzer-store=region -analyzer-constraints=basic -verify %s
// RUN: clang-cc -analyze -checker-cfref -analyzer-store=region -analyzer-constraints=range -verify %s

typedef struct CGColorSpace *CGColorSpaceRef;
extern CGColorSpaceRef CGColorSpaceCreateDeviceRGB(void);
extern CGColorSpaceRef CGColorSpaceRetain(CGColorSpaceRef space);
extern void CGColorSpaceRelease(CGColorSpaceRef space);

void f() {
  CGColorSpaceRef X = CGColorSpaceCreateDeviceRGB(); // expected-warning{{leak}}
  CGColorSpaceRetain(X);
}

void fb() {
  CGColorSpaceRef X = CGColorSpaceCreateDeviceRGB();
  CGColorSpaceRetain(X);
  CGColorSpaceRelease(X);
  CGColorSpaceRelease(X);  // no-warning
}
