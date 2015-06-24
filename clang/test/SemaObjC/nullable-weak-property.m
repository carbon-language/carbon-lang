// RUN: %clang_cc1 -fobjc-arc -fobjc-runtime-has-weak -Wnullable-to-nonnull-conversion %s -verify


// rdar://19985330
@interface NSObject @end

@class NSFoo;
void foo (NSFoo * _Nonnull);

@interface NSBar : NSObject
@property(weak) NSFoo *property1;
@end

@implementation NSBar 
- (void) Meth {
   foo (self.property1); // expected-warning {{implicit conversion from nullable pointer 'NSFoo * _Nullable' to non-nullable pointer type 'NSFoo * _Nonnull'}}
}
@end
