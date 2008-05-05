// RUN: clang -checker-cfref -verify -fobjc-gc %s

#include <CoreFoundation/CFDate.h>
#include <Foundation/NSDate.h>
#include <Foundation/NSZone.h>

CFAbsoluteTime f1() {
  CFAbsoluteTime t = CFAbsoluteTimeGetCurrent();
  CFDateRef date = CFDateCreate(NULL, t);
  CFRetain(date);
  [NSMakeCollectable(date) release];
  CFDateGetAbsoluteTime(date); // no-warning
  CFRelease(date);
  t = CFDateGetAbsoluteTime(date);   // expected-warning{{Reference-counted object is used after it is released.}}
  return t;
}

