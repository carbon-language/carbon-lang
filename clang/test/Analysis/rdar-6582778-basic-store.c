// RUN: %clang_cc1 -analyze -analyzer-checker=core,core.experimental -analyzer-store=basic -verify %s

typedef const void * CFTypeRef;
typedef double CFTimeInterval;
typedef CFTimeInterval CFAbsoluteTime;
typedef const struct __CFAllocator * CFAllocatorRef;
typedef const struct __CFDate * CFDateRef;

extern CFDateRef CFDateCreate(CFAllocatorRef allocator, CFAbsoluteTime at);
CFAbsoluteTime CFAbsoluteTimeGetCurrent(void);

void f(void) {
  CFAbsoluteTime t = CFAbsoluteTimeGetCurrent();
  CFTypeRef vals[] = { CFDateCreate(0, t) }; // no-warning
}

CFTypeRef global;

void g(void) {
  CFAbsoluteTime t = CFAbsoluteTimeGetCurrent();
  global = CFDateCreate(0, t); // no-warning
}
