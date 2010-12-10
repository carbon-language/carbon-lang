// RUN: %clang_cc1  -fsyntax-only -verify %s
// rdar://7766184

@interface Foo @end

@interface Foo ()
  @property (readonly) int bar;
@end

void FUNC () {
    Foo *foo;
    foo.bar = 0; // expected-error {{assigning to property with 'readonly' attribute not allowed}}
}

// rdar://8747333
@class NSObject;

@interface rdar8747333  {
@private
    NSObject *_bar;
    NSObject *_baz;
}
- (NSObject *)baz;
@end

@interface rdar8747333 ()
- (NSObject *)bar;
@end

@interface rdar8747333 ()
@property (readwrite, assign) NSObject *bar;
@property (readwrite, assign) NSObject *baz;
@end

@implementation rdar8747333
@synthesize bar = _bar;
@synthesize baz = _baz;
@end

