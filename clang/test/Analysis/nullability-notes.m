// RUN: %clang_analyze_cc1 -fblocks -analyzer-checker=core,nullability.NullPassedToNonnull,nullability.NullReturnedFromNonnull,nullability.NullablePassedToNonnull,nullability.NullableReturnedFromNonnull,nullability.NullableDereferenced -analyzer-output=text -verify %s

#include "Inputs/system-header-simulator-for-nullability.h"

void takesNonnull(NSObject *_Nonnull y);

@interface ClassWithProperties: NSObject
@property(copy, nullable) NSObject *x;
-(void) method;
@end;
@implementation ClassWithProperties
-(void) method {
  // no-crash
  NSObject *x = self.x; // expected-note{{Nullability 'nullable' is inferred}}
  takesNonnull(x); // expected-warning{{Nullable pointer is passed to a callee that requires a non-null 1st parameter}}
                   // expected-note@-1{{Nullable pointer is passed to a callee that requires a non-null 1st parameter}}
}
@end
