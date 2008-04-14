// RUN: clang -checker-cfref -verify %s

#include <CoreFoundation/CFDate.h>

CFAbsoluteTime f1() {
  CFAbsoluteTime t = CFAbsoluteTimeGetCurrent();
  CFDateRef date = CFDateCreate(NULL, t);
  CFRetain(date);
  CFRelease(date);
  CFDateGetAbsoluteTime(date);
  CFRelease(date);
  t = CFDateGetAbsoluteTime(date);   // expected-warning{{Reference-counted object is used after it is released.}}
  return t;
}

