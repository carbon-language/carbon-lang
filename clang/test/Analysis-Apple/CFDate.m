// RUN: clang -checker-cfref -verify %s

#include <CoreFoundation/CFDate.h>
#include <Foundation/NSDate.h>

CFAbsoluteTime f1() {
  CFAbsoluteTime t = CFAbsoluteTimeGetCurrent();
  CFDateRef date = CFDateCreate(NULL, t);
  CFRetain(date);
  CFRelease(date);
  CFDateGetAbsoluteTime(date); // no-warning
  CFRelease(date);
  t = CFDateGetAbsoluteTime(date);   // expected-warning{{Reference-counted object is used after it is released.}}
  return t;
}

CFAbsoluteTime f2() {
  CFAbsoluteTime t = CFAbsoluteTimeGetCurrent();
  CFDateRef date = CFDateCreate(NULL, t);  
  [((NSDate*) date) retain];
  CFRelease(date);
  CFDateGetAbsoluteTime(date); // no-warning
  [((NSDate*) date) release];
  t = CFDateGetAbsoluteTime(date);   // expected-warning{{Reference-counted object is used after it is released.}}
  return t;
}

