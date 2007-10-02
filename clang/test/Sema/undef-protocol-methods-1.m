@protocol P1
- (void) P1proto; // expected-warning {{method definition for 'P1proto' not found}}
+ (void) ClsP1Proto; // expected-warning {{method definition for 'ClsP1Proto' not found}}
- (void) DefP1proto;
@end
@protocol P2
- (void) P2proto;  // expected-warning {{method definition for 'P2proto' not found}}
+ (void) ClsP2Proto; // expected-warning {{method definition for 'ClsP2Proto' not found}}
@end

@protocol P3<P2>
- (void) P3proto; // expected-warning {{method definition for 'P3proto' not found}}
+ (void) ClsP3Proto; // expected-warning {{method definition for 'ClsP3Proto' not found}}
+ (void) DefClsP3Proto;
@end

@protocol PROTO<P1, P3>
- (void) meth;			// expected-warning {{method definition for 'meth' not found
- (void) meth : (int) arg1;	// expected-warning {{method definition for 'meth:' not found
+ (void) cls_meth : (int) arg1; // expected-warning {{method definition for 'cls_meth:' not found
@end

@interface INTF <PROTO>
@end

@implementation INTF
- (void) DefP1proto{}

+ (void) DefClsP3Proto{}

@end // expected-warning {{ncomplete implementation of class 'INTF'}}
