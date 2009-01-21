// RUN: clang -analyze -checker-cfref -analyzer-store-basic -verify %s &&
// RUN: clang -analyze -checker-cfref -analyzer-store-region -verify %s

typedef struct CGColorSpace *CGColorSpaceRef;
extern CGColorSpaceRef CGColorSpaceCreateDeviceRGB(void);
extern CGColorSpaceRef CGColorSpaceRetain(CGColorSpaceRef space);
extern void CGColorSpaceRelease(CGColorSpaceRef space);

void f() {
  CGColorSpaceRef X = CGColorSpaceCreateDeviceRGB();
  CGColorSpaceRetain(X);  // expected-warning{{leak}}
}

void fb() {
  CGColorSpaceRef X = CGColorSpaceCreateDeviceRGB();
  CGColorSpaceRetain(X);
  CGColorSpaceRelease(X);
  CGColorSpaceRelease(X);  // no-warning
}
