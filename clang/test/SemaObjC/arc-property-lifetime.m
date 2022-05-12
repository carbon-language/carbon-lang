// RUN: %clang_cc1 -triple x86_64-apple-darwin11 -fobjc-runtime-has-weak -fsyntax-only -fobjc-arc -verify -Wno-objc-root-class %s
// rdar://9340606

@interface Foo {
@public
    id __unsafe_unretained x; // expected-error {{existing instance variable 'x' for strong property 'x' may not be __unsafe_unretained}}
    id __weak y; // expected-error {{existing instance variable 'y' for strong property 'y' may not be __weak}}
    id __autoreleasing z; // expected-error {{instance variables cannot have __autoreleasing ownership}}
}
@property(strong) id x; // expected-note {{property declared here}}
@property(strong) id y; // expected-note {{property declared here}}
@property(strong) id z;
@end

@implementation Foo
@synthesize x; // expected-note {{property synthesized here}}
@synthesize y; // expected-note {{property synthesized here}}
@synthesize z; // suppressed
@end

@interface Bar {
@public
    id __unsafe_unretained x; // expected-error {{existing instance variable 'x' for strong property 'x' may not be __unsafe_unretained}}
    id __weak y; // expected-error {{existing instance variable 'y' for strong property 'y' may not be __weak}}
    id __autoreleasing z; // expected-error {{instance variables cannot have __autoreleasing ownership}}
}
@property(retain) id x; // expected-note {{property declared here}}
@property(retain) id y; // expected-note {{property declared here}}
@property(retain) id z;
@end

@implementation Bar
@synthesize x; // expected-note {{property synthesized here}}
@synthesize y; // expected-note {{property synthesized here}}
@synthesize z; // suppressed
@end

@interface Bas {
@public
    id __unsafe_unretained x; // expected-error {{existing instance variable 'x' for strong property 'x' may not be __unsafe_unretained}}
    id __weak y; // expected-error {{existing instance variable 'y' for strong property 'y' may not be __weak}}
    id __autoreleasing z; // expected-error {{instance variables cannot have __autoreleasing ownership}}
}
@property(copy) id x; // expected-note {{property declared here}}
@property(copy) id y; // expected-note {{property declared here}} 
@property(copy) id z;
@end

@implementation Bas
@synthesize x; // expected-note {{property synthesized here}}
@synthesize y; // expected-note {{property synthesized here}}
@synthesize z; // suppressed
@end

@interface Bat 
@property(strong) __unsafe_unretained id x; // expected-error {{strong property 'x' may not also be declared __unsafe_unretained}}
@property(strong) __autoreleasing id z; // expected-error {{strong property 'z' may not also be declared __autoreleasing}}
@end

@interface Bau 
@property(retain) __unsafe_unretained id x; // expected-error {{strong property 'x' may not also be declared __unsafe_unretained}}
@property(retain) __autoreleasing id z; // expected-error {{strong property 'z' may not also be declared __autoreleasing}}
@end

@interface Bav 
@property(copy) __unsafe_unretained id x; // expected-error {{strong property 'x' may not also be declared __unsafe_unretained}}
@property(copy) __autoreleasing id z; // expected-error {{strong property 'z' may not also be declared __autoreleasing}}
@end

// rdar://9341593
@interface Gorf  {
   id __unsafe_unretained x;
   id y; // expected-error {{existing instance variable 'y' for property 'y' with assign attribute must be __unsafe_unretained}}
}
@property(assign) id __unsafe_unretained x;
@property(assign) id y; // expected-note {{property declared here}}
@property(assign) id z;
@end

@implementation Gorf
@synthesize x;
@synthesize y; // expected-note {{property synthesized here}}
@synthesize z;
@end

@interface Gorf2  {
   id __unsafe_unretained x;
   id y; // expected-error {{existing instance variable 'y' for property 'y' with unsafe_unretained attribute must be __unsafe_unretained}}
}
@property(unsafe_unretained) id __unsafe_unretained x;
@property(unsafe_unretained) id y; // expected-note {{property declared here}}
@property(unsafe_unretained) id z;
@end

@implementation Gorf2
@synthesize x;
@synthesize y; // expected-note {{property synthesized here}}
@synthesize z;
@end

// rdar://9355230
@interface I {
  char _isAutosaving;
}
@property char isAutosaving;

@end

@implementation I
@synthesize isAutosaving = _isAutosaving;
@end

// rdar://10239594
// Test for 'Class' properties being unretained.
@interface MyClass {
@private
    Class _controllerClass;
    id _controllerId;
}
@property (copy) Class controllerClass;
@property (copy) id controllerId;
@end

@implementation MyClass
@synthesize controllerClass = _controllerClass;
@synthesize controllerId = _controllerId;
@end

// rdar://10630891
@interface UIView @end
@class UIColor;

@interface UIView(UIViewRendering)
@property(nonatomic,copy) UIColor *backgroundColor;
@end

@interface UILabel : UIView
@end

@interface MyView 
@property (strong) UILabel *label;
@end

@interface MyView2 : MyView @end

@implementation MyView2
- (void)foo {
  super.label.backgroundColor = 0;
}
@end

// rdar://10694932
@interface Baz 
@property  id prop;
@property  __strong id strong_prop;
@property  (strong) id strong_attr_prop;
@property  (strong) __strong id really_strong_attr_prop;
+ (id) alloc;
- (id) init;
- (id) implicit;
- (void) setImplicit : (id) arg; 
@end

void foo(Baz *f) {
        f.prop = [[Baz alloc] init];
        f.strong_prop = [[Baz alloc] init];
        f.strong_attr_prop = [[Baz alloc] init];
        f.really_strong_attr_prop = [[Baz alloc] init];
        f.implicit = [[Baz alloc] init];
}

// rdar://11253688
@interface Boom 
{
  const void * innerPointerIvar __attribute__((objc_returns_inner_pointer)); // expected-error {{'objc_returns_inner_pointer' attribute only applies to Objective-C methods and Objective-C properties}}
}
@property (readonly) Boom * NotInnerPointer __attribute__((objc_returns_inner_pointer)); // expected-warning {{'objc_returns_inner_pointer' attribute only applies to properties that return a non-retainable pointer}}
- (Boom *) NotInnerPointerMethod __attribute__((objc_returns_inner_pointer)); // expected-warning {{'objc_returns_inner_pointer' attribute only applies to methods that return a non-retainable pointer}}
@property (readonly) const void * innerPointer __attribute__((objc_returns_inner_pointer));
@end

@interface Foo2 {
  id _prop; // expected-error {{existing instance variable '_prop' for property 'prop' with assign attribute must be __unsafe_unretained}}
}
@property (nonatomic, assign) id prop; // expected-note {{property declared here}}
@end

@implementation Foo2
@end

// rdar://13885083
@interface NSObject 
-(id)init;
@end

typedef char BOOL;
@interface Test13885083 : NSObject

@property (nonatomic, assign) BOOL retain; // expected-error {{ARC forbids synthesis of 'retain'}}

-(id)init;

@end

@implementation Test13885083
-(id) init
{
  self = [super init];
  return self;
}
@end

