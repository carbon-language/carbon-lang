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

// CHECK: define {{.*}}testCleanupVoid
// CHECK: icmp ne {{.*}}!dbg ![[SKIP1:[0-9]+]]
// CHECK: store i32 0, i32* {{.*}}, !dbg ![[RET8:[0-9]+]]
// CHECK: @objc_storeStrong{{.*}}, !dbg ![[ARC8:[0-9]+]]
// CHECK: ret {{.*}} !dbg ![[RET8]]

typedef signed char BOOL;

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
  // CHECK: ![[ARC1]] = !{i32 [[@LINE+1]], i32 0, ![[TESTNOSIDEEFFECT]], null}
  return 1; // Return expression
  // CHECK: ![[RET1]] = !{i32 [[@LINE+1]], i32 0, !{{.*}}, null}
}           // Cleanup + Ret

- (int)testNoCleanup {
  // CHECK: ![[RET2]] = !{i32 [[@LINE+1]], i32 0, !{{.*}}, null}
  return 1;
}

- (int)testSideEffect:(NSString *)foo {
  // CHECK: ![[MSG3]] = !{i32 [[@LINE+1]], i32 0, !{{.*}}, null}
  return [self testNoSideEffect :foo];
  // CHECK: ![[RET3]] = !{i32 [[@LINE+1]], i32 0, !{{.*}}, null}
}

- (int)testMultiline:(NSString *)foo {
  // CHECK: ![[MSG4]] = !{i32 [[@LINE+1]], i32 0, !{{.*}}, null}
  int r = [self testSideEffect :foo];
  // CHECK: ![[EXP4]] = !{i32 [[@LINE+1]], i32 0, !{{.*}}, null}
  return r;
  // CHECK: ![[RET4]] = !{i32 [[@LINE+1]], i32 0, !{{.*}}, null}
}

- (void)testVoid:(NSString *)foo {
  // CHECK: ![[ARC5]] = !{i32 [[@LINE+1]], i32 0, !{{.*}}, null}
  return;
  // CHECK: ![[RET5]] = !{i32 [[@LINE+1]], i32 0, !{{.*}}, null}
}

- (void)testVoidNoReturn:(NSString *)foo {
  // CHECK: ![[MSG6]] = !{i32 [[@LINE+1]], i32 0, !{{.*}}, null}
  [self testVoid :foo];
  // CHECK: ![[RET6]] = !{i32 [[@LINE+1]], i32 0, !{{.*}}, null}
}

- (int)testNoCleanupSideEffect {
  // CHECK: ![[MSG7]] = !{i32 [[@LINE+1]], i32 0, !{{.*}}, null}
  [self testVoid :@"foo"];
  // CHECK: ![[RET7]] = !{i32 [[@LINE+1]], i32 0, !{{.*}}, null}
  return 1;
}

- (void)testCleanupVoid:(BOOL)skip withDelegate: (AppDelegate *) delegate {
  static BOOL skip_all;
  // CHECK: ![[SKIP1]] = !{i32 [[@LINE+1]], i32 0,
  if (!skip_all) {
    if (!skip) {
      return;
    }
    NSString *s = @"bar";
    if (!skip) {
      [delegate testVoid :s];
    }
  }
  // CHECK: ![[RET8]] = !{i32 [[@LINE+2]], i32 0,
  // CHECK: ![[ARC8]] = !{i32 [[@LINE+1]], i32 0,
}


@end


int main(int argc, const char** argv) {
  AppDelegate *o = [[AppDelegate alloc] init];
  return [o testMultiline :@"foo"];
}
