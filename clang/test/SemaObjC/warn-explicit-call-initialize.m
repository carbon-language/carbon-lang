// RUN: %clang_cc1  -fsyntax-only  -triple x86_64-apple-darwin10 -verify %s
// rdar://16628028

@interface NSObject
+ (void)initialize; // expected-note {{method 'initialize' declared here}}
@end

@interface I : NSObject 
+ (void)initialize; // expected-note {{method 'initialize' declared here}}
@end

@implementation I
- (void) Meth { 
  [I initialize];     // expected-warning {{explicit call to +initialize results in duplicate call to +initialize}} 
  [NSObject initialize]; // expected-warning {{explicit call to +initialize results in duplicate call to +initialize}}
}
+ (void)initialize {}
@end

