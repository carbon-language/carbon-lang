// RUN: %clang_cc1 -fsyntax-only -verify %s

@protocol P1
- (void) P1proto; // expected-warning {{method in protocol not implemented [-Wprotocol]}}
+ (void) ClsP1Proto;   // expected-warning {{method in protocol not implemented [-Wprotocol]}}
- (void) DefP1proto;
@end
@protocol P2
- (void) P2proto;   // expected-warning {{method in protocol not implemented [-Wprotocol]}}
+ (void) ClsP2Proto;  // expected-warning {{method in protocol not implemented [-Wprotocol]}}
@end

@protocol P3<P2>
- (void) P3proto;  // expected-warning {{method in protocol not implemented [-Wprotocol]}}
+ (void) ClsP3Proto;  // expected-warning {{method in protocol not implemented [-Wprotocol]}}
+ (void) DefClsP3Proto;
@end

@protocol PROTO<P1, P3>
- (void) meth;		 // expected-warning {{method in protocol not implemented [-Wprotocol]}}
- (void) meth : (int) arg1;  // expected-warning {{method in protocol not implemented [-Wprotocol]}}
+ (void) cls_meth : (int) arg1;  // expected-warning {{method in protocol not implemented [-Wprotocol]}}
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
