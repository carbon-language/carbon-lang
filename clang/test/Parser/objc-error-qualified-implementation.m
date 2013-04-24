// RUN: %clang_cc1 -fsyntax-only -Wno-objc-root-class -verify %s
// rdar://12233858

@protocol P
@end

@interface I @end

@implementation I<P> @end // expected-error {{@implementation declaration can not be protocol qualified}}

@interface J < P,P >
@end


@implementation J < P,P > // expected-error {{@implementation declaration can not be protocol qualified}}
@end

@interface K @end

@implementation K <P // expected-error {{@implementation declaration can not be protocol qualified}}
@end // expected-error {{expected '>'}}
