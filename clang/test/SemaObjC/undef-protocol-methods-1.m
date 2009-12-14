// RUN: clang -cc1 -fsyntax-only -verify %s

@protocol P1
- (void) P1proto;
+ (void) ClsP1Proto;  
- (void) DefP1proto;
@end
@protocol P2
- (void) P2proto;  
+ (void) ClsP2Proto; 
@end

@protocol P3<P2>
- (void) P3proto; 
+ (void) ClsP3Proto; 
+ (void) DefClsP3Proto;
@end

@protocol PROTO<P1, P3>
- (void) meth;		
- (void) meth : (int) arg1; 
+ (void) cls_meth : (int) arg1; 
@end

@interface INTF <PROTO>
@end

@implementation INTF   // expected-warning {{incomplete implementation}} \
                          expected-warning {{method definition for 'meth' not found}} \
                          expected-warning {{method definition for 'meth:' not found}} \
                          expected-warning {{method definition for 'cls_meth:' not found}} \
                          expected-warning {{method definition for 'P3proto' not found}} \
                          expected-warning {{method definition for 'ClsP3Proto' not found}} \
                          expected-warning {{method definition for 'P2proto' not found}} \
                          expected-warning {{method definition for 'ClsP2Proto' not found}} \
                          expected-warning {{method definition for 'ClsP1Proto' not found}} \
                          expected-warning {{method definition for 'P1proto' not found}}
- (void) DefP1proto{}

+ (void) DefClsP3Proto{}

@end
