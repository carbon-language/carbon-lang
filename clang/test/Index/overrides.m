
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
// CHECK: overrides.m:12:9: ObjCInstanceMethodDecl=protoMethod:12:9 [Overrides @3:9]
// CHECK: overrides.m:22:9: ObjCInstanceMethodDecl=method:22:9 [Overrides @16:9]
// CHECK: overrides.m:23:9: ObjCInstanceMethodDecl=protoMethod:23:9 [Overrides @12:9, @8:9, @32:9, @17:9]
// CHECK: overrides.m:27:9: ObjCInstanceMethodDecl=method:27:9 (Definition) [Overrides @16:9]
// CHECK: overrides.m:28:9: ObjCClassMethodDecl=methodWithParam::28:9 (Definition) [Overrides @18:9]
// CHECK: overrides.m:32:9: ObjCInstanceMethodDecl=protoMethod:32:9 [Overrides @8:9]
// CHECK: overrides.m:36:9: ObjCInstanceMethodDecl=protoMethod:36:9 [Overrides @12:9, @8:9, @32:9, @17:9]
