// RUN: %clang_cc1 -fsyntax-only -verify %s
// rdar: //7824372

@interface A
-(void) im0;

@implementation A // expected-error {{missing @end}}
@end

@interface B {
}

@implementation B // expected-error {{missing @end}}
@end

@interface C
@property int P;

@implementation C // expected-error 2 {{missing @end}}
