// RUN: %clang -g -std=c++11 -S -emit-llvm %s -o - | FileCheck %s
// PR19864
int main() {
    int v[] = {13, 21, 8, 3, 34, 1, 5, 2};
    int a = 0, b = 0;
    for (int x : v)
      if (x >= 3)
        ++b;     // CHECK: add nsw{{.*}}, 1
      else if (x >= 0)
        ++a;    // CHECK: add nsw{{.*}}, 1
    // The continuation block if the if statement should not share the
    // location of the ++a statement. Having it point to the end of
    // the condition is not ideal either, but it's less missleading.

    // CHECK: br label
    // CHECK: br label
    // CHECK: br label {{.*}}, !dbg ![[DBG:.*]]
    // CHECK: ![[DBG]] = metadata !{i32 [[@LINE-11]], i32 0, metadata !{{.*}}, null}

}
