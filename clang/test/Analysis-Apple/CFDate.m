// RUN: clang -checker-cfref -verify %s

#include <CoreFoundation/CFDate.h>
#include <Foundation/NSDate.h>
#include <Foundation/NSCalendarDate.h>
#include <Foundation/NSString.h>

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


NSDate* global_x;

  // Test to see if we supresss an error when we store the pointer
  // to a global.
  
CFAbsoluteTime f3() {
  CFAbsoluteTime t = CFAbsoluteTimeGetCurrent();
  CFDateRef date = CFDateCreate(NULL, t);  
  [((NSDate*) date) retain];
  CFRelease(date);
  CFDateGetAbsoluteTime(date); // no-warning
  global_x = (NSDate*) date;  
  [((NSDate*) date) release];
  t = CFDateGetAbsoluteTime(date);   // no-warning
  return t;
}

// Test to see if we supresss an error when we store the pointer
// to a struct.

struct foo {
  NSDate* f;
};

CFAbsoluteTime f4() {
  struct foo x;
  
  CFAbsoluteTime t = CFAbsoluteTimeGetCurrent();
  CFDateRef date = CFDateCreate(NULL, t);  
  [((NSDate*) date) retain];
  CFRelease(date);
  CFDateGetAbsoluteTime(date); // no-warning
  x.f = (NSDate*) date;  
  [((NSDate*) date) release];
  t = CFDateGetAbsoluteTime(date);   // no-warning
  return t;
}

// Test a leak.

CFAbsoluteTime f5(int x) {  
  CFAbsoluteTime t = CFAbsoluteTimeGetCurrent();
  CFDateRef date = CFDateCreate(NULL, t);
  
  if (x)
    CFRelease(date);

  return t; // expected-warning{{leak}}
}

// Test a leak involving the return.

CFDateRef f6(int x) {  
  CFDateRef date = CFDateCreate(NULL, CFAbsoluteTimeGetCurrent());
  CFRetain(date);
  return date; // expected-warning{{leak}}
}

// Test a leak involving an overwrite.

CFDateRef f7() {
  CFDateRef date = CFDateCreate(NULL, CFAbsoluteTimeGetCurrent());
  CFRetain(date);
  date = CFDateCreate(NULL, CFAbsoluteTimeGetCurrent()); //expected-warning{{leak}}
  return date;
}

NSDate* f8(int x) {
 
  NSDate* date = [NSDate date];
  
  if (x) [date retain];
  
  return date; // expected-warning{{leak}}
}

NSDate* f9(int x) {
  
  NSDate* date = [NSDate dateWithString:@"2001-03-24 10:45:32 +0600"];
  
  if (x) [date retain];
  
  return date; // expected-warning{{leak}}
}

