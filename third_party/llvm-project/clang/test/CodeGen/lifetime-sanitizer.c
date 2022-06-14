// RUN: %clang -Xclang -no-opaque-pointers -target x86_64-linux-gnu -S -emit-llvm -o - -O0 \
// RUN:     -Xclang -disable-llvm-passes %s | FileCheck %s -check-prefix=CHECK-O0
// RUN: %clang -Xclang -no-opaque-pointers -target x86_64-linux-gnu -S -emit-llvm -o - -O0 \
// RUN:     -fsanitize=address -fsanitize-address-use-after-scope \
// RUN:     -Xclang -disable-llvm-passes %s | FileCheck %s -check-prefix=LIFETIME
// RUN: %clang -Xclang -no-opaque-pointers -target x86_64-linux-gnu -S -emit-llvm -o - -O0 \
// RUN:     -fsanitize=memory -Xclang -disable-llvm-passes %s | \
// RUN:     FileCheck %s -check-prefix=LIFETIME
// RUN: %clang -Xclang -no-opaque-pointers -target aarch64-linux-gnu -S -emit-llvm -o - -O0 \
// RUN:     -fsanitize=hwaddress -Xclang -disable-llvm-passes %s | \
// RUN:     FileCheck %s -check-prefix=LIFETIME

extern int bar(char *A, int n);

// CHECK-O0-NOT: @llvm.lifetime.start
int foo(int n) {
  if (n) {
    // LIFETIME: @llvm.lifetime.start.p0i8(i64 10, i8* {{.*}})
    char A[10];
    return bar(A, 1);
    // LIFETIME: @llvm.lifetime.end.p0i8(i64 10, i8* {{.*}})
  } else {
    // LIFETIME: @llvm.lifetime.start.p0i8(i64 20, i8* {{.*}})
    char A[20];
    return bar(A, 2);
    // LIFETIME: @llvm.lifetime.end.p0i8(i64 20, i8* {{.*}})
  }
}
