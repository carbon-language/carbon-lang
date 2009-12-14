// RUN: clang -cc1 -fsyntax-only -verify %s

@protocol MyProto1 
@optional
- (void) FOO;
@optional
- (void) FOO;
@required 
- (void) REQ;
@optional
@end

@protocol  MyProto2 <MyProto1>
- (void) FOO2;
@optional
- (void) FOO3;
@end
