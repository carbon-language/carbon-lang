// RUN: %clang_cc1 -analyze -analyzer-checker=alpha.osx.cocoa.DirectIvarAssignment -fobjc-default-synthesize-properties -Wno-objc-root-class -verify -fblocks %s

@interface MyClass;
@end
@interface TestProperty {
  MyClass *_Z;
  id _nonSynth;
}

  @property (assign, nonatomic) MyClass* A; // explicitely synthesized, not implemented, non-default ivar name

  @property (assign) MyClass* X;  // automatically synthesized, not implemented

  @property (assign, nonatomic) MyClass* Y; // automatically synthesized, implemented

  @property (assign, nonatomic) MyClass* Z; // non synthesized ivar, implemented setter
  @property (readonly) id nonSynth;  // non synthesized, explicitly implemented to return ivar with expected name
  
  - (id) initWithPtr:(MyClass*) value;
  - (id) myInitWithPtr:(MyClass*) value;
  - (void) someMethod: (MyClass*)In;
@end

@implementation TestProperty
  @synthesize A = __A;
  
  - (id) initWithPtr: (MyClass*) value {
    _Y = value; // no-warning
    return self;
  }

  - (id) myInitWithPtr: (MyClass*) value {
    _Y = value; // no-warning
    return self;
  }
  
  - (void) setY:(MyClass*) NewValue {
    _Y = NewValue; // no-warning
  }

  - (void) setZ:(MyClass*) NewValue {
    _Z = NewValue; // no-warning
  }

  - (id)nonSynth {
      return _nonSynth;
  }

  - (void) someMethod: (MyClass*)In {
    (__A) = In; // expected-warning {{Direct assignment to an instance variable backing a property; use the setter instead}}
    _X = In; // expected-warning {{Direct assignment to an instance variable backing a property; use the setter instead}}
    _Y = In; // expected-warning {{Direct assignment to an instance variable backing a property; use the setter instead}}
    _Z = In; // expected-warning {{Direct assignment to an instance variable backing a property; use the setter instead}}
    _nonSynth = 0; // expected-warning {{Direct assignment to an instance variable backing a property; use the setter instead}}
  }
@end