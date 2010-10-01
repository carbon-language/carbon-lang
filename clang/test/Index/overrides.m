
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

// RUN: c-index-test -test-load-source local %s | FileCheck %s
// CHECK: overrides.m:12:1: ObjCInstanceMethodDecl=protoMethod:12:1 [Overrides @3:1] Extent=[12:1 - 12:21]
// CHECK: overrides.m:21:1: ObjCInstanceMethodDecl=method:21:1 [Overrides @16:1] Extent=[21:1 - 21:16]
// CHECK: overrides.m:22:1: ObjCInstanceMethodDecl=protoMethod:22:1 [Overrides @12:1, @8:1] Extent=[22:1 - 22:21]
// CHECK: overrides.m:26:1: ObjCInstanceMethodDecl=method:26:1 (Definition) [Overrides @21:1] Extent=[26:1 - 26:19]
// CHECK: overrides.m:27:1: ObjCClassMethodDecl=methodWithParam::27:1 (Definition) [Overrides @17:1] Extent=[27:1 - 27:39]
