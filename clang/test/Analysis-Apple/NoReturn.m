// RUN: clang -checker-simple -verify %s
// RUN: clang -checker-cfref -verify %s


#include <Foundation/NSException.h>
#include <Foundation/NSString.h>

int f1(int *x, NSString* s) {
  
  if (x) ++x;
  
  [NSException raise:@"Blah" format:[NSString stringWithFormat:@"Blah %@", s]];
  
  return *x; // no-warning
}

int f2(int *x, ...) {
  
  if (x) ++x;
  va_list alist;
  va_start(alist, x);
  
  [NSException raise:@"Blah" format:@"Blah %@" arguments:alist];
  
  return *x; // no-warning
}

int f3(int* x) {
  
  if (x) ++x;
  
  [[NSException exceptionWithName:@"My Exception" reason:@"Want to test exceptions." userInfo:nil] raise];

  return *x; // no-warning
}

