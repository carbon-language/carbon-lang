// RUN: %clang_cc1 -verify %s

@protocol MyProto1 
@optional
- (void) FOO;
@optional
- (void) FOO;
@optional 
- (void) REQ;
@optional
@end

@interface  MyProto2 <MyProto1>
- (void) FOO2;
- (void) FOO3;
@end

@implementation MyProto2
- (void) FOO2{}
- (void) FOO3{}
@end
