// RUN: %clang_cc1 -S -emit-llvm -o - -fsanitize=address,use-after-scope %s \
// RUN:     | FileCheck %s -check-prefix=USE-AFTER-SCOPE
// RUN: %clang_cc1 -S -emit-llvm -o - -fsanitize=address %s \
// RUN:     | FileCheck %s -check-prefix=ADDRESS-ONLY

extern int bar(char *A, int n);

// ADDRESS-ONLY-NOT: @llvm.lifetime.start
int foo (int n) {
  if (n) {
    // USE-AFTER-SCOPE: @llvm.lifetime.start(i64 10, i8* {{.*}})
    char A[10];
    return bar(A, 1);
    // USE-AFTER-SCOPE: @llvm.lifetime.end(i64 10, i8* {{.*}})
  } else {
    // USE-AFTER-SCOPE: @llvm.lifetime.start(i64 20, i8* {{.*}})
    char A[20];
    return bar(A, 2);
    // USE-AFTER-SCOPE: @llvm.lifetime.end(i64 20, i8* {{.*}})
  }
}

