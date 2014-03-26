// RUN: %clang_cc1 -emit-llvm %s -o - | FileCheck %s

char buffer[32] = "This is a largely unused buffer";

// __builtin___clear_cache always maps to @llvm.clear_cache, but what
// each back-end produces is different, and this is tested in LLVM

int main() {
  __builtin___clear_cache(buffer, buffer+32);
// CHECK: @llvm.clear_cache(i8* getelementptr {{.*}}, i8* getelementptr {{.*}} (i8* getelementptr {{.*}} 32))
  return 0;
}
