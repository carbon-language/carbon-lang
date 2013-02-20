// RUN: %clang_cc1 -Oz -emit-llvm %s -o - | FileCheck %s -check-prefix=Oz
// RUN: %clang_cc1 -O0 -emit-llvm %s -o - | FileCheck %s -check-prefix=OTHER
// RUN: %clang_cc1 -O1 -emit-llvm %s -o - | FileCheck %s -check-prefix=OTHER
// RUN: %clang_cc1 -O2 -emit-llvm %s -o - | FileCheck %s -check-prefix=OTHER
// RUN: %clang_cc1 -O3 -emit-llvm %s -o - | FileCheck %s -check-prefix=OTHER
// RUN: %clang_cc1 -Os -emit-llvm %s -o - | FileCheck %s -check-prefix=OTHER
// Check that we set the minsize attribute on each function
// when Oz optimization level is set.

int test1() {
  return 42;
// Oz: @{{.*}}test1{{.*}}#0
// Oz: ret
// OTHER: @{{.*}}test1
// OTHER-NOT: #1
// OTHER: ret
}

int test2() {
  return 42;
// Oz: @{{.*}}test2{{.*}}#0
// Oz: ret
// OTHER: @{{.*}}test2
// OTHER-NOT: #1
// OTHER: ret
}

__attribute__((minsize))
int test3() {
  return 42;
// Oz: @{{.*}}test3{{.*}}#0
// OTHER: @{{.*}}test3{{.*}}#1
}

// Check that the minsize attribute is well propagated through
// template instantiation

template<typename T>
__attribute__((minsize))
void test4(T arg) {
  return;
}

template
void test4<int>(int arg);
// Oz: define{{.*}}void @{{.*}}test4
// Oz: #0
// OTHER: define{{.*}}void @{{.*}}test4
// OTHER: #1

template
void test4<float>(float arg);
// Oz: define{{.*}}void @{{.*}}test4
// Oz: #0
// OTHER: define{{.*}}void @{{.*}}test4
// OTHER: #1

template<typename T>
void test5(T arg) {
  return;
}

template
void test5<int>(int arg);
// Oz: define{{.*}}void @{{.*}}test5
// Oz: #0
// OTHER: define{{.*}}void @{{.*}}test5
// OTHER-NOT: define{{.*}}void @{{.*}}test5{{.*}}#1

template
void test5<float>(float arg);
// Oz: define{{.*}}void @{{.*}}test5
// Oz: #0
// OTHER: define{{.*}}void @{{.*}}test5
// OTHER-NOT: define{{.*}}void @{{.*}}test5{{.*}}#1

// Oz: attributes #0 = { minsize nounwind optsize readnone "target-features"={{.*}} }

// OTHER: attributes #0 = { nounwind {{.*}}"target-features"={{.*}} }
// OTHER: attributes #1 = { minsize nounwind {{.*}}"target-features"={{.*}} }
