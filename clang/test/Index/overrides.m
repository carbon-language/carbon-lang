
@protocol P1
- (void)protoMethod;
- (void)protoMethodWithParam:(int)param;
@end

@protocol P3
- (void)protoMethod;
@end

@protocol P2 <P1>
- (void)protoMethod;
@end

@interface A
- (void)method;
- (void)protoMethod;
+ (void)methodWithParam:(int)param;
@end

@interface B : A <P2, P3>
- (void)method;
- (void)protoMethod;
@end

@implementation B
- (void)method { }
+ (void)methodWithParam:(int)param { }
@end

@protocol P4 <P3>
- (void)protoMethod;
@end

@interface B(cat) <P4>
- (void)protoMethod;
@end

// RUN: c-index-test -test-load-source local %s | FileCheck %s
// CHECK: overrides.m:12:1: ObjCInstanceMethodDecl=protoMethod:12:1 [Overrides @3:1]
// CHECK: overrides.m:22:1: ObjCInstanceMethodDecl=method:22:1 [Overrides @16:1]
// CHECK: overrides.m:23:1: ObjCInstanceMethodDecl=protoMethod:23:1 [Overrides @12:1, @8:1, @32:1, @17:1]
// CHECK: overrides.m:27:1: ObjCInstanceMethodDecl=method:27:1 (Definition) [Overrides @16:1]
// CHECK: overrides.m:28:1: ObjCClassMethodDecl=methodWithParam::28:1 (Definition) [Overrides @18:1]
// CHECK: overrides.m:32:1: ObjCInstanceMethodDecl=protoMethod:32:1 [Overrides @8:1]
// CHECK: overrides.m:36:1: ObjCInstanceMethodDecl=protoMethod:36:1 [Overrides @12:1, @8:1, @32:1, @17:1]
