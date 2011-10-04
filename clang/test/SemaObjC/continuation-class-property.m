// RUN: %clang_cc1  -fsyntax-only -verify %s
// radar 7509234

@protocol Foo
@property (readonly, copy) id foos;
@end

@interface Bar <Foo> {
}

@end

@interface Baz  <Foo> {
}
@end

@interface Bar ()
@property (readwrite, copy) id foos;
@end

@interface Baz ()
@property (readwrite, copy) id foos;
@end


// rdar://10142679
@class NSString;

typedef struct {
  float width;
  float length;
} NSRect;

@interface MyClass  {
}
@property (readonly) NSRect foo; // expected-note {{property declared here}}
@property (readonly, strong) NSString *bar; // expected-note {{property declared here}}
@end

@interface MyClass ()
@property (readwrite) NSString *foo; // expected-warning {{type of property 'NSString *' in continuation class does not match property type in primary class}}
@property (readwrite, strong) NSRect bar; // expected-warning {{type of property 'NSRect' in continuation class does not match property type in primary class}}
@end
