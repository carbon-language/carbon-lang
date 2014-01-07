// RUN: %clang_cc1 -emit-llvm -fblocks -fobjc-arc -g -triple x86_64-apple-darwin10 %s -o - | FileCheck %s

// Legend: EXP = Return expression, RET = ret instruction

// CHECK: define {{.*}}testNoSideEffect
// CHECK: call void @objc_storeStrong{{.*}}
// CHECK: call void @objc_storeStrong{{.*}} !dbg ![[ARC1:[0-9]+]]
// CHECK: ret {{.*}} !dbg ![[RET1:[0-9]+]]

// CHECK: define {{.*}}testNoCleanup
// CHECK: ret {{.*}} !dbg ![[RET2:[0-9]+]]

// CHECK: define {{.*}}testSideEffect
// CHECK: @objc_msgSend{{.*}} !dbg ![[MSG3:[0-9]+]]
// CHECK: ret {{.*}} !dbg ![[RET3:[0-9]+]]

// CHECK: define {{.*}}testMultiline
// CHECK: @objc_msgSend{{.*}} !dbg ![[MSG4:[0-9]+]]
// CHECK: load{{.*}} !dbg ![[EXP4:[0-9]+]]
// CHECK: ret {{.*}} !dbg ![[RET4:[0-9]+]]

// CHECK: define {{.*}}testVoid
// CHECK: call void @objc_storeStrong{{.*}}
// CHECK: call void @objc_storeStrong{{.*}} !dbg ![[ARC5:[0-9]+]]
// CHECK: ret {{.*}} !dbg ![[RET5:[0-9]+]]

// CHECK: define {{.*}}testVoidNoReturn
// CHECK: @objc_msgSend{{.*}} !dbg ![[MSG6:[0-9]+]]
// CHECK: ret {{.*}} !dbg ![[RET6:[0-9]+]]

// CHECK: define {{.*}}testNoCleanupSideEffect
// CHECK: @objc_msgSend{{.*}} !dbg ![[MSG7:[0-9]+]]
// CHECK: ret {{.*}} !dbg ![[RET7:[0-9]+]]


@interface NSObject
+ (id)alloc;
- (id)init;
- (id)retain;
@end

@class NSString;

@interface AppDelegate : NSObject

@end

@implementation AppDelegate : NSObject

// CHECK: ![[TESTNOSIDEEFFECT:.*]] = {{.*}}[ DW_TAG_subprogram ] [line [[@LINE+1]]] [local] [def] [-[AppDelegate testNoSideEffect:]]
- (int)testNoSideEffect:(NSString *)foo {
  int x = 1;
  // CHECK: ![[ARC1]] = metadata !{i32 [[@LINE+1]], i32 0, metadata ![[TESTNOSIDEEFFECT]], null}
  return 1; // Return expression
  // CHECK: ![[RET1]] = metadata !{i32 [[@LINE+1]], i32 0, metadata !{{.*}}, null}
}           // Cleanup + Ret

- (int)testNoCleanup {
  // CHECK: ![[RET2]] = metadata !{i32 [[@LINE+1]], i32 0, metadata !{{.*}}, null}
  return 1;
}

- (int)testSideEffect:(NSString *)foo {
  // CHECK: ![[MSG3]] = metadata !{i32 [[@LINE+1]], i32 0, metadata !{{.*}}, null}
  return [self testNoSideEffect :foo];
  // CHECK: ![[RET3]] = metadata !{i32 [[@LINE+1]], i32 0, metadata !{{.*}}, null}
}

- (int)testMultiline:(NSString *)foo {
  // CHECK: ![[MSG4]] = metadata !{i32 [[@LINE+1]], i32 0, metadata !{{.*}}, null}
  int r = [self testSideEffect :foo];
  // CHECK: ![[EXP4]] = metadata !{i32 [[@LINE+1]], i32 0, metadata !{{.*}}, null}
  return r;
  // CHECK: ![[RET4]] = metadata !{i32 [[@LINE+1]], i32 0, metadata !{{.*}}, null}
}

- (void)testVoid:(NSString *)foo {
  // CHECK: ![[ARC5]] = metadata !{i32 [[@LINE+1]], i32 0, metadata !{{.*}}, null}
  return;
  // CHECK: ![[RET5]] = metadata !{i32 [[@LINE+1]], i32 0, metadata !{{.*}}, null}
}

- (void)testVoidNoReturn:(NSString *)foo {
  // CHECK: ![[MSG6]] = metadata !{i32 [[@LINE+1]], i32 0, metadata !{{.*}}, null}
  [self testVoid :foo];
  // CHECK: ![[RET6]] = metadata !{i32 [[@LINE+1]], i32 0, metadata !{{.*}}, null}
}

- (int)testNoCleanupSideEffect {
  // CHECK: ![[MSG7]] = metadata !{i32 [[@LINE+1]], i32 0, metadata !{{.*}}, null}
  [self testVoid :@"foo"];
  // CHECK: ![[RET7]] = metadata !{i32 [[@LINE+1]], i32 0, metadata !{{.*}}, null}
  return 1;
}


@end


int main(int argc, const char** argv) {
  AppDelegate *o = [[AppDelegate alloc] init];
  return [o testMultiline :@"foo"];
}
