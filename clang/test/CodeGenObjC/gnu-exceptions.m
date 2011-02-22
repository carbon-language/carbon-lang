// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -emit-llvm -fexceptions -fobjc-exceptions -fgnu-runtime -o - %s | FileCheck %s

void opaque(void);
void log(int i);

@class C;

// CHECK: define void @test0() {
void test0() {
  @try {
    // CHECK: invoke void @opaque()
    opaque();

    // CHECK: call void @log(i32 1)

  } @catch (C *c) {
    // CHECK: call i8* @llvm.eh.exception()
    // CHECK: call i32 (i8*, i8*, ...)* @llvm.eh.selector({{.*}} @__gnu_objc_personality_v0
    // CHECK: br i1

    // CHECK: call void @log(i32 0)

    // CHECK: call void @objc_exception_throw

    log(0);
  }

  log(1);
}
