// RUN: %clang_cc1 -fsyntax-only -verify %s

@protocol P1
- (void) P1proto;  // expected-note {{method declared here}}
+ (void) ClsP1Proto;    // expected-note {{method declared here}}
- (void) DefP1proto;
@end
@protocol P2
- (void) P2proto;   // expected-note {{method declared here}}
+ (void) ClsP2Proto;  // expected-note {{method declared here}}
@end

@protocol P3<P2>
- (void) P3proto;   // expected-note {{method declared here}}
+ (void) ClsP3Proto;   // expected-note {{method declared here}}
+ (void) DefClsP3Proto;
@end

@protocol PROTO<P1, P3>
- (void) meth;		  // expected-note {{method declared here}}
- (void) meth : (int) arg1;   // expected-note {{method declared here}}
+ (void) cls_meth : (int) arg1;   // expected-note {{method declared here}}
@end

@interface INTF <PROTO> // expected-note 3 {{required for direct or indirect protocol 'PROTO'}} \
			// expected-note 2 {{required for direct or indirect protocol 'P1'}} \
			// expected-note 2 {{required for direct or indirect protocol 'P3'}} \
			// expected-note 2 {{required for direct or indirect protocol 'P2'}}
@end

@implementation INTF   // expected-warning {{incomplete implementation}} \
                       // expected-warning 9 {{method in protocol not implemented [-Wprotocol}}
- (void) DefP1proto{}

+ (void) DefClsP3Proto{}

@end
