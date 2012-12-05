// RUN: %clang_cc1 -analyze -analyzer-checker=alpha.osx.cocoa.DirectIvarAssignmentForAnnotatedFunctions -fobjc-default-synthesize-properties -verify -fblocks %s

typedef signed char BOOL;
@protocol NSObject  - (BOOL)isEqual:(id)object; @end
@interface NSObject <NSObject> {}
+(id)alloc;
-(id)init;
-(id)autorelease;
-(id)copy;
-(id)retain;
@end

@interface MyClass;
@end

@interface AnnotatedClass :NSObject {
}
  - (void) someMethod: (MyClass*)In __attribute__((annotate("objc_no_direct_instance_variable_assignmemt")));
  - (void) someMethodNotAnnaotated: (MyClass*)In;
@end


@interface TestProperty :AnnotatedClass {
  MyClass *_Z;
  id _nonSynth;
}

  @property (assign, nonatomic) MyClass* A; // explicitely synthesized, not implemented, non-default ivar name

  @property (assign) MyClass* X;  // automatically synthesized, not implemented

  @property (assign, nonatomic) MyClass* Y; // automatically synthesized, implemented

  @property (assign, nonatomic) MyClass* Z; // non synthesized ivar, implemented setter
  @property (readonly) id nonSynth;  // non synthesized, explicitly implemented to return ivar with expected name
  @end

@implementation TestProperty
  @synthesize A = __A;
  
  - (void) someMethod: (MyClass*)In {
    (__A) = In; // expected-warning {{Direct assignment to an instance variable backing a property; use the setter instead}}
    _X = In; // expected-warning {{Direct assignment to an instance variable backing a property; use the setter instead}}
    _Y = In; // expected-warning {{Direct assignment to an instance variable backing a property; use the setter instead}}
    _Z = In; // expected-warning {{Direct assignment to an instance variable backing a property; use the setter instead}}
    _nonSynth = 0; // expected-warning {{Direct assignment to an instance variable backing a property; use the setter instead}}
  }
  - (void) someMethodNotAnnaotated: (MyClass*)In {
    (__A) = In; 
    _X = In; // no-warning
    _Y = In; // no-warning
    _Z = In; // no-warning
    _nonSynth = 0; // no-warning
  }

@end