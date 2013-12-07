// RUN: %clang_cc1 -fsyntax-only -x objective-c -verify -fobjc-arc -Wno-objc-root-class %s
// rdar://15499111

typedef struct __attribute__((objc_bridge_related(NSColor,colorWithCGColor:,CGColor))) CGColor *CGColorRef; // expected-note 5 {{declared here}}
typedef struct __attribute__((objc_bridge_related(NSColor,,CGColor1))) CGColor1 *CGColorRef1; // expected-note 3 {{declared here}}
typedef struct __attribute__((objc_bridge_related(NSColor,,))) CGColor2 *CGColorRef2; // expected-note 2 {{declared here}}

@interface NSColor // expected-note 10 {{declared here}}
+ (NSColor *)colorWithCGColor:(CGColorRef)cgColor;
- (CGColorRef)CGColor;
- (CGColorRef1)CGColor1;
@end

@interface NSTextField 
- (void)setBackgroundColor:(NSColor *)color;
- (NSColor *)backgroundColor;
@end

void foo(NSColor*);

NSColor * Test1(NSTextField *textField, CGColorRef newColor) {
  foo(newColor); // expected-error {{'CGColorRef' (aka 'struct CGColor *') must be explicitly converted to 'NSColor *', use 'colorWithCGColor:' method for this conversion}} \
		 // expected-error {{implicit conversion of C pointer type 'CGColorRef' (aka 'struct CGColor *') to Objective-C pointer type 'NSColor *' requires a bridged cast}} \
		 // expected-note {{use __bridge to convert directly (no change in ownership)}} \
                 // expected-note {{use __bridge_transfer to transfer ownership of a +1 'CGColorRef' (aka 'struct CGColor *') into ARC}}
  textField.backgroundColor = newColor; // expected-error {{'CGColorRef' (aka 'struct CGColor *') must be explicitly converted to 'NSColor *__strong', use 'colorWithCGColor:' method for this conversion}} \
					// expected-error {{implicit conversion of C pointer type 'CGColorRef' (aka 'struct CGColor *') to Objective-C pointer type 'NSColor *' requires a bridged cast}} \
                                        // expected-note {{use __bridge to convert directly (no change in ownership)}} \
					// expected-note {{use __bridge_transfer to transfer ownership of a +1 'CGColorRef' (aka 'struct CGColor *') into ARC}}
  return newColor; // expected-error {{'CGColorRef' (aka 'struct CGColor *') must be explicitly converted to 'NSColor *', use 'colorWithCGColor:' method for this conversion}} \
					// expected-error {{implicit conversion of C pointer type 'CGColorRef' (aka 'struct CGColor *') to Objective-C pointer type 'NSColor *' requires a bridged cast}} \
                                        // expected-note {{use __bridge to convert directly (no change in ownership)}} \
					// expected-note {{use __bridge_transfer to transfer ownership of a +1 'CGColorRef' (aka 'struct CGColor *') into ARC}}
}

NSColor * Test2(NSTextField *textField, CGColorRef1 newColor) {
  foo(newColor); // expected-error {{'CGColorRef1' (aka 'struct CGColor1 *') must be explicitly converted to 'NSColor *', define and then use a singular class method in NSColor for this conversion}} \
                 // expected-error {{implicit conversion of C pointer type 'CGColorRef1' (aka 'struct CGColor1 *') to Objective-C pointer type 'NSColor *' requires a bridged cast}} \
		 // expected-note {{use __bridge to convert directly (no change in ownership)}} \ 
		 // expected-note {{use __bridge_transfer to transfer ownership of a +1 'CGColorRef1' (aka 'struct CGColor1 *') into ARC}}
  textField.backgroundColor = newColor;  // expected-error {{'CGColorRef1' (aka 'struct CGColor1 *') must be explicitly converted to 'NSColor *__strong', define and then use a singular class method in NSColor for this conversion}} \
                 // expected-error {{implicit conversion of C pointer type 'CGColorRef1' (aka 'struct CGColor1 *') to Objective-C pointer type 'NSColor *' requires a bridged cast}}  \
		 // expected-note {{use __bridge to convert directly (no change in ownership)}} \ 
		 // expected-note {{use __bridge_transfer to transfer ownership of a +1 'CGColorRef1' (aka 'struct CGColor1 *') into ARC}}
  return newColor;  // expected-error {{'CGColorRef1' (aka 'struct CGColor1 *') must be explicitly converted to 'NSColor *', define and then use a singular class method in NSColor for this conversion}} \
                 // expected-error {{implicit conversion of C pointer type 'CGColorRef1' (aka 'struct CGColor1 *') to Objective-C pointer type 'NSColor *' requires a bridged cast}} \
		 // expected-note {{use __bridge to convert directly (no change in ownership)}} \ 
		 // expected-note {{use __bridge_transfer to transfer ownership of a +1 'CGColorRef1' (aka 'struct CGColor1 *') into ARC}}
}

CGColorRef Test3(NSTextField *textField, CGColorRef newColor) {
  newColor = textField.backgroundColor; // expected-error {{'NSColor *' must be explicitly converted to 'CGColorRef' (aka 'struct CGColor *'), use 'CGColor' method for this conversion}} \
                                        // expected-error {{implicit conversion of Objective-C pointer type 'NSColor *' to C pointer type 'CGColorRef' (aka 'struct CGColor *') requires a bridged cast}} \
				        // expected-note {{use __bridge to convert directly (no change in ownership)}} \ 
				        // expected-note {{use __bridge_retained to make an ARC object available as a +1 'CGColorRef' (aka 'struct CGColor *')}}
  return textField.backgroundColor; // expected-error {{'NSColor *' must be explicitly converted to 'CGColorRef' (aka 'struct CGColor *'), use 'CGColor' method for this conversion}} \
                                        // expected-error {{implicit conversion of Objective-C pointer type 'NSColor *' to C pointer type 'CGColorRef' (aka 'struct CGColor *') requires a bridged cast}} \
					// expected-note {{use __bridge to convert directly (no change in ownership)}} \ 
				        // expected-note {{use __bridge_retained to make an ARC object available as a +1 'CGColorRef' (aka 'struct CGColor *')}}
}

CGColorRef2 Test4(NSTextField *textField, CGColorRef2 newColor) {
  newColor = textField.backgroundColor; // expected-error {{'NSColor *' must be explicitly converted to 'CGColorRef2' (aka 'struct CGColor2 *'), define and then use an instance method in NSColor for this conversion}} \
                                        // expected-error {{implicit conversion of Objective-C pointer type 'NSColor *' to C pointer type 'CGColorRef2' (aka 'struct CGColor2 *') requires a bridged cast}} \
					// expected-note {{use __bridge to convert directly (no change in ownership)}} \ 
					// expected-note {{use __bridge_retained to make an ARC object available as a +1 'CGColorRef2' (aka 'struct CGColor2 *')}}
  return textField.backgroundColor; // expected-error {{'NSColor *' must be explicitly converted to 'CGColorRef2' (aka 'struct CGColor2 *'), define and then use an instance method in NSColor for this conversion}} \
                                        // expected-error {{implicit conversion of Objective-C pointer type 'NSColor *' to C pointer type 'CGColorRef2' (aka 'struct CGColor2 *') requires a bridged cast}} \
					// expected-note {{use __bridge to convert directly (no change in ownership)}} \ 
					// expected-note {{use __bridge_retained to make an ARC object available as a +1 'CGColorRef2' (aka 'struct CGColor2 *')}}
}
