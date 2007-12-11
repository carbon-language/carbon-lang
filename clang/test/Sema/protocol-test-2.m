// RUN: clang -fsyntax-only -verify %s

@interface INTF1 @end

@protocol p1,p2,p3;

@protocol p1;

@protocol PROTO1
- (INTF1<p1>*) meth;
@end

@protocol PROTO2<p1> // expected-warning {{cannot find protocol definition for 'p1', referenced by 'PROTO2'}}
@end

@protocol p1 @end

@protocol PROTO<p1>
@end

@protocol PROTO<p1>	// expected-error {{duplicate protocol declaration of 'PROTO'}}
@end

@protocol PROTO3<p1, p1>
@end

@protocol p2 <p1>
@end

@protocol PROTO4 <p1, p2, PROTO, PROTO3, p3> // expected-warning {{cannot find protocol definition for 'p3', referenced by 'PROTO4'}}
@end
