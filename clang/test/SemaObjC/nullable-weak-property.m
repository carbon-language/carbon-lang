// RUN: %clang_cc1 -fobjc-arc -fobjc-runtime-has-weak -Wnullable-to-nonnull-conversion %s -verify


// rdar://19985330
@interface NSObject @end

@class NSFoo;
void foo (NSFoo * __nonnull);

@interface NSBar : NSObject
@property(weak) NSFoo *property1;
@end

@implementation NSBar 
- (void) Meth {
   foo (self.property1); // expected-warning {{implicit conversion from nullable pointer 'NSFoo * __nullable' to non-nullable pointer type 'NSFoo * __nonnull'}}
}
@end
