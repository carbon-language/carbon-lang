// RUN: %clang_cc1  -fsyntax-only -triple thumbv6-apple-ios3.0 -verify -Wno-objc-root-class %s
// RUN: %clang_cc1 -x objective-c++ -fsyntax-only -triple thumbv6-apple-ios3.0 -verify -Wno-objc-root-class %s
// rdar://12324295

typedef signed char BOOL;

@protocol P
@property(nonatomic,assign) id ptarget __attribute__((availability(ios,introduced=2.0,deprecated=3.0))); // expected-note {{property 'ptarget' is declared deprecated here}} expected-note {{'ptarget' has been explicitly marked deprecated here}}
@end

@protocol P1<P>
- (void)setPtarget:(id)arg;
@end


@interface UITableViewCell<P1>
@property(nonatomic,assign) id target __attribute__((availability(ios,introduced=2.0,deprecated=3.0))); // expected-note {{property 'target' is declared deprecated here}} expected-note {{'setTarget:' has been explicitly marked deprecated here}}
@end

@interface PSTableCell : UITableViewCell
 - (void)setTarget:(id)target;
@end

@interface UITableViewCell(UIDeprecated)
@property(nonatomic,assign) id dep_target  __attribute__((availability(ios,introduced=2.0,deprecated=3.0))); // expected-note 2 {{'dep_target' has been explicitly marked deprecated here}} \
                                                                                    // expected-note 4 {{property 'dep_target' is declared deprecated here}} \
                                                                                    // expected-note 2 {{'setDep_target:' has been explicitly marked deprecated here}}
@end

@implementation PSTableCell
- (void)setTarget:(id)target {};
- (void)setPtarget:(id)val {};
- (void) Meth {
  [self setTarget: (id)0]; // no-warning
  [self setDep_target: [self dep_target]]; // expected-warning {{'dep_target' is deprecated: first deprecated in iOS 3.0}} \
                                           // expected-warning {{'setDep_target:' is deprecated: first deprecated in iOS 3.0}}
					   
  [self setPtarget: (id)0]; // no-warning
}
@end

@implementation UITableViewCell
@synthesize target;
@synthesize ptarget;
- (void)setPtarget:(id)val {};
- (void)setTarget:(id)target {};
- (void) Meth {
  [self setTarget: (id)0]; // expected-warning {{'setTarget:' is deprecated: first deprecated in iOS 3.0}}
  [self setDep_target: [self dep_target]]; // expected-warning {{'dep_target' is deprecated: first deprecated in iOS 3.0}} \
                                           // expected-warning {{'setDep_target:' is deprecated: first deprecated in iOS 3.0}}
					   
  [self setPtarget: (id)0]; // no-warning
}
@end


@interface CustomAccessorNames
@property(getter=isEnabled,assign) BOOL enabled __attribute__((availability(ios,introduced=2.0,deprecated=3.0))); // expected-note {{'isEnabled' has been explicitly marked deprecated here}} expected-note {{property 'enabled' is declared deprecated here}}

@property(setter=setNewDelegate:,assign) id delegate __attribute__((availability(ios,introduced=2.0,deprecated=3.0))); // expected-note {{'setNewDelegate:' has been explicitly marked deprecated here}} expected-note {{property 'delegate' is declared deprecated here}}
@end

void testCustomAccessorNames(CustomAccessorNames *obj) {
  if ([obj isEnabled]) // expected-warning {{'isEnabled' is deprecated: first deprecated in iOS 3.0}}
    [obj setNewDelegate:0]; // expected-warning {{'setNewDelegate:' is deprecated: first deprecated in iOS 3.0}}
}


@interface ProtocolInCategory
@end

@interface ProtocolInCategory (TheCategory) <P1>
- (id)ptarget;
@end

id useDeprecatedProperty(ProtocolInCategory *obj, id<P> obj2, int flag) {
  if (flag)
    return [obj ptarget];  // no-warning
  return [obj2 ptarget];   // expected-warning {{'ptarget' is deprecated: first deprecated in iOS 3.0}}
}

// rdar://15951801
@interface Foo
{
  int _x;
}
@property(nonatomic,readonly) int x;
- (void)setX:(int)x __attribute__ ((deprecated)); // expected-note 2 {{'setX:' has been explicitly marked deprecated here}}
- (int)x __attribute__ ((unavailable)); // expected-note {{'x' has been explicitly marked unavailable here}}
@end

@implementation Foo
- (void)setX:(int)x {
	_x = x;
}
- (int)x {
  return _x;
}
@end

void testUserAccessorAttributes(Foo *foo) {
        [foo setX:5678]; // expected-warning {{'setX:' is deprecated}}
	foo.x = foo.x; // expected-error {{property access is using 'x' method which is unavailable}} \
		       // expected-warning {{property access is using 'setX:' method which is deprecated}}
}
