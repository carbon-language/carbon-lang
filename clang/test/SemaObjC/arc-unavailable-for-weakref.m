// RUN: %clang_cc1 -triple x86_64-apple-darwin11 -fobjc-runtime-has-weak -fsyntax-only -fobjc-arc -verify -Wno-objc-root-class %s
// RUN: %clang_cc1 -triple x86_64-apple-darwin11 -fobjc-runtime-has-weak -fsyntax-only -fobjc-weak -verify -Wno-objc-root-class %s
// rdar://9693477

__attribute__((objc_arc_weak_reference_unavailable))
@interface NSOptOut1072  // expected-note {{class is declared here}}
@end

@interface sub : NSOptOut1072 @end // expected-note 2 {{class is declared here}}

int main(void) {
  __weak sub *w2; // expected-error {{class is incompatible with __weak references}}

  __weak NSOptOut1072 *ns1; // expected-error {{class is incompatible with __weak references}}

  id obj;

  ns1 = (__weak sub *)obj; // expected-error {{assignment of a weak-unavailable object to a __weak object}} \
                           // expected-error {{class is incompatible with __weak references}} \
                           // expected-error {{explicit ownership qualifier on cast result has no effect}}
}

// rdar://9732636
__attribute__((objc_arc_weak_reference_unavailable))
@interface NOWEAK
+ (id) new;
@end

NOWEAK * Test1(void) {
  NOWEAK * strong1 = [NOWEAK new];
  __weak id weak1;
  weak1 = strong1; // expected-error {{assignment of a weak-unavailable object to a __weak object}}

  __weak id weak2 = strong1; // expected-error {{assignment of a weak-unavailable object to a __weak object}}
  return (__weak id)strong1; // expected-error {{cast of weak-unavailable object of type 'NOWEAK *' to a __weak object of type '__weak id'}} \
                             // expected-error {{explicit ownership qualifier on cast result has no effect}}
}

@protocol P @end
@protocol P1 @end

NOWEAK<P, P1> * Test2(void) {
  NOWEAK<P, P1> * strong1 = 0;
  __weak id<P> weak1;
  weak1 = strong1; // expected-error {{assignment of a weak-unavailable object to a __weak object}}

  __weak id<P> weak2 = strong1; // expected-error {{assignment of a weak-unavailable object to a __weak object}}
  return (__weak id<P>)strong1; // expected-error {{cast of weak-unavailable object of type 'NOWEAK<P,P1> *' to a __weak object of type '__weak id<P>'}} \
                                // expected-error {{explicit ownership qualifier on cast result has no effect}}
}

// rdar://10535245
__attribute__((objc_arc_weak_reference_unavailable))
@interface NSFont
@end

@interface I
{
}
@property (weak) NSFont *font; // expected-error {{synthesizing __weak instance variable of type 'NSFont *', which does not support weak references}}
@end

@implementation I // expected-note {{when implemented by class I}}
@synthesize font = _font;
@end

// rdar://13676793
@protocol MyProtocol
@property (weak) NSFont *font; // expected-error {{synthesizing __weak instance variable of type 'NSFont *', which does not support weak references}}
@end

@interface I1 <MyProtocol>
@end

@implementation I1 // expected-note {{when implemented by class I1}}
@synthesize font = _font;
@end

@interface Super
@property (weak) NSFont *font;  // expected-error {{synthesizing __weak instance variable of type 'NSFont *', which does not support weak references}}
@end


@interface I2 : Super
@end

@implementation I2 // expected-note {{when implemented by class I2}}
@synthesize font = _font;
@end

__attribute__((objc_arc_weak_reference_unavailable(1)))	// expected-error {{'objc_arc_weak_reference_unavailable' attribute takes no arguments}}
@interface I3
@end

int I4 __attribute__((objc_arc_weak_reference_unavailable)); // expected-error {{'objc_arc_weak_reference_unavailable' attribute only applies to Objective-C interfaces}}
