// RUN: %clang_cc1 %s -triple=x86_64-apple-darwin10 -emit-llvm -o - | FileCheck %s
// rdar: // 8027844

// CHECK: call void @llvm.memset

int main() {
    id foo;
    for (id a in foo) {
    }
}
