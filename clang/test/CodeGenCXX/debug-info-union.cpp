// RUN: %clang_cc1 -emit-llvm -g -triple x86_64-apple-darwin -std=c++11 %s -o - | FileCheck %s

union E {
  int a;
  float b;
  int bb() { return a;}
  float aa() { return b;}
  E() { a = 0; }
};

E e;

// CHECK: !MDCompositeType(tag: DW_TAG_union_type, name: "E"
// CHECK-SAME:             line: 3
// CHECK-SAME:             size: 32, align: 32
// CHECK-NOT:              offset:
// CHECK-SAME:             {{$}}
// CHECK: !MDSubprogram(name: "bb"{{.*}}, line: 6
// CHECK: !MDSubprogram(name: "aa"{{.*}}, line: 7
// CHECK: !MDSubprogram(name: "E"{{.*}}, line: 8
