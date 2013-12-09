// RUN: %clang_cc1 -fsyntax-only -x objective-c -verify -Wno-objc-root-class %s
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
  foo(newColor); // expected-error {{'CGColorRef' (aka 'struct CGColor *') must be explicitly converted to 'NSColor *'; use '+colorWithCGColor:' method for this conversion}}
  textField.backgroundColor = newColor; // expected-error {{'CGColorRef' (aka 'struct CGColor *') must be explicitly converted to 'NSColor *'; use '+colorWithCGColor:' method for this conversion}}
  return newColor; // expected-error {{'CGColorRef' (aka 'struct CGColor *') must be explicitly converted to 'NSColor *'; use '+colorWithCGColor:' method for this conversion}}
}

NSColor * Test2(NSTextField *textField, CGColorRef1 newColor) {
  foo(newColor); // expected-error {{you can't convert 'CGColorRef1' (aka 'struct CGColor1 *') to 'NSColor *', without using an existing class method for this conversion}}
  textField.backgroundColor = newColor;  // expected-error {{you can't convert 'CGColorRef1' (aka 'struct CGColor1 *') to 'NSColor *', without using an existing class method for this conversion}}
  return newColor;  // expected-error {{you can't convert 'CGColorRef1' (aka 'struct CGColor1 *') to 'NSColor *', without using an existing class method for this conversion}}
}

CGColorRef Test3(NSTextField *textField, CGColorRef newColor) {
  newColor = textField.backgroundColor; // expected-error {{'NSColor *' must be explicitly converted to 'CGColorRef' (aka 'struct CGColor *'); use '-CGColor' method for this conversion}}
  return textField.backgroundColor; // expected-error {{'NSColor *' must be explicitly converted to 'CGColorRef' (aka 'struct CGColor *'); use '-CGColor' method for this conversion}}
}

CGColorRef2 Test4(NSTextField *textField, CGColorRef2 newColor) {
  newColor = textField.backgroundColor; // expected-error {{you can't convert 'NSColor *' to 'CGColorRef2' (aka 'struct CGColor2 *'), without using an existing instance method for this conversion}}
  return textField.backgroundColor; // expected-error {{you can't convert 'NSColor *' to 'CGColorRef2' (aka 'struct CGColor2 *'), without using an existing instance method for this conversion}}
}
