// RUN: %clang_cc1  -fsyntax-only -verify %s
// rdar://7822210

@interface A @end

@implementation A @end // expected-note {{class implementation is declared here}}

@interface A () // expected-error {{cannot declare class extension for 'A' after class implementation}}
-(void) im0;
@end

