// RUN: clang -verify %s

@protocol P
@end

@interface INTF<P>
- (void)IMeth;
 - (void) Meth;
@end

@implementation INTF
- (void)IMeth { [(id<P>)self Meth]; }  // expected-warning {{method '-Meth' not found in protocol (return type defaults to 'id')}}
- (void) Meth {}
@end
