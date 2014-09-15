// RUN: %clang_cc1 %s -fexceptions -fseh-exceptions -emit-llvm -triple x86_64-w64-windows-gnu -o - | FileCheck %s

extern "C" void foo();
extern "C" void bar();

struct Cleanup {
  ~Cleanup() {
    bar();
  }
};

extern "C" void test() {
  Cleanup x;
  foo();
}

// CHECK: define void @test()
// CHECK: invoke void @foo()
// CHECK: landingpad { i8*, i32 } personality i8* bitcast (i32 (...)* @__gxx_personality_seh0 to i8*)
