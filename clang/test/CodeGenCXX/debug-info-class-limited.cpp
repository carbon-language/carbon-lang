// RUN: %clang -emit-llvm -g -S %s -o - | FileCheck %s

namespace PR16214_1 {
// CHECK: [[PR16214_1:![0-9]*]] = {{.*}} [ DW_TAG_namespace ] [PR16214_1]
// CHECK: = metadata !{i32 {{[0-9]*}}, metadata !{{[0-9]*}}, metadata [[PR16214_1]], {{.*}} ; [ DW_TAG_structure_type ] [foo] {{.*}} [def]
struct foo {
  int i;
};

typedef foo bar;

bar *a;
bar b;
}

namespace test1 {
struct foo {
  int i;
};

foo *bar(foo *a) {
  foo *b = new foo(*a);
  return b;
}
}

namespace test2 {
struct foo {
  int i;
};

extern int bar(foo *a);
int baz(foo *a) {
  return bar(a);
}
}
