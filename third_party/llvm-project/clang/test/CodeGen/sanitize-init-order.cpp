// RUN: %clang_cc1 -fsanitize=address -emit-llvm -o - %s | FileCheck %s

// Test ignorelist functionality.
// RUN: echo "src:%s=init" | sed -e 's/\\/\\\\/g' > %t-file.ignorelist
// RUN: echo "type:PODWithCtorAndDtor=init" > %t-type.ignorelist
// RUN: echo "type:NS::PODWithCtor=init" >> %t-type.ignorelist
// RUN: %clang_cc1 -fsanitize=address -fsanitize-ignorelist=%t-file.ignorelist -emit-llvm -o - %s | FileCheck %s --check-prefix=IGNORELIST
// RUN: %clang_cc1 -fsanitize=address -fsanitize-ignorelist=%t-type.ignorelist -emit-llvm -o - %s | FileCheck %s --check-prefix=IGNORELIST

struct PODStruct {
  int x;
};
PODStruct s1;

struct PODWithDtor {
  ~PODWithDtor() { }
  int x;
};
PODWithDtor s2;

struct PODWithCtorAndDtor {
  PODWithCtorAndDtor() { }
  ~PODWithCtorAndDtor() { }
  int x;
};
PODWithCtorAndDtor s3;

namespace NS {
class PODWithCtor {
public:
  PODWithCtor() {}
};

const volatile PODWithCtor array[5][5];
}

// Check that ASan init-order checking ignores structs with trivial default
// constructor.
// CHECK: !llvm.asan.globals = !{![[GLOB_1:[0-9]+]], ![[GLOB_2:[0-9]+]], ![[GLOB_3:[0-9]+]], ![[GLOB_4:[0-9]+]]
// CHECK: ![[GLOB_1]] = !{%struct.PODStruct* {{.*}}, i1 false, i1 false}
// CHECK: ![[GLOB_2]] = !{%struct.PODWithDtor* {{.*}}, i1 false, i1 false}
// CHECK: ![[GLOB_3]] = !{%struct.PODWithCtorAndDtor* {{.*}}, i1 true, i1 false}
// CHECK: ![[GLOB_4]] = !{{{.*}}class.NS::PODWithCtor{{.*}}, i1 true, i1 false}

// IGNORELIST: !llvm.asan.globals = !{![[GLOB_1:[0-9]+]], ![[GLOB_2:[0-9]+]], ![[GLOB_3:[0-9]+]], ![[GLOB_4:[0-9]+]]}
// IGNORELIST: ![[GLOB_1]] = !{%struct.PODStruct* {{.*}}, i1 false, i1 false}
// IGNORELIST: ![[GLOB_2]] = !{%struct.PODWithDtor* {{.*}}, i1 false, i1 false}
// IGNORELIST: ![[GLOB_3]] = !{%struct.PODWithCtorAndDtor* {{.*}}, i1 false, i1 false}
// IGNORELIST: ![[GLOB_4]] = !{{{.*}}class.NS::PODWithCtor{{.*}}, i1 false, i1 false}
