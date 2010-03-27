// RUN: %clang_cc1 -fsyntax-only -verify %s

@protocol P1
- (void) P1proto; // expected-note {{method definition for 'P1proto' not found}}
+ (void) ClsP1Proto;   // expected-note {{method definition for 'ClsP1Proto' not found}}
- (void) DefP1proto;
@end
@protocol P2
- (void) P2proto;   // expected-note {{method definition for 'P2proto' not found}}
+ (void) ClsP2Proto;  // expected-note {{method definition for 'ClsP2Proto' not found}}
@end

@protocol P3<P2>
- (void) P3proto;  // expected-note {{method definition for 'P3proto' not found}}
+ (void) ClsP3Proto;  // expected-note {{method definition for 'ClsP3Proto' not found}}
+ (void) DefClsP3Proto;
@end

@protocol PROTO<P1, P3>
- (void) meth;		 // expected-note {{method definition for 'meth' not found}}
- (void) meth : (int) arg1;  // expected-note {{method definition for 'meth:' not found}}
+ (void) cls_meth : (int) arg1;  // expected-note {{method definition for 'cls_meth:' not found}}
@end

@interface INTF <PROTO> // expected-note 3 {{required for direct or indirect protocol 'PROTO'}} \
			// expected-note 2 {{required for direct or indirect protocol 'P1'}} \
			// expected-note 2 {{required for direct or indirect protocol 'P3'}} \
			// expected-note 2 {{required for direct or indirect protocol 'P2'}}
@end

@implementation INTF   // expected-warning {{incomplete implementation}} 
- (void) DefP1proto{}

+ (void) DefClsP3Proto{}

@end
