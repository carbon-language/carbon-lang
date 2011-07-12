// RUN: %clang_cc1 -fsyntax-only -verify %s
// rdar://9740328
// XFAIL: *

@protocol P1;

@interface NSObject
@end

@interface A : NSObject
@property (assign) NSObject<P1> *prop;
@end

@protocol P2 <P1>
@end

@interface B : A
@property (assign) NSObject<P2> *prop;
@end

