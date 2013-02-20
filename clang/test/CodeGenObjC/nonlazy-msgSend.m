// RUN: %clang_cc1 -triple x86_64-apple-darwin9 -fobjc-runtime=macosx-fragile-10.5 -emit-llvm %s -o - | FileCheck %s

// CHECK: declare i8* @objc_msgSend(i8*, i8*, ...) #1
void f0(id x) {
  [x foo];
}

// CHECK: attributes #0 = { nounwind "target-features"={{.*}} }
// CHECK: attributes #1 = { nonlazybind }
