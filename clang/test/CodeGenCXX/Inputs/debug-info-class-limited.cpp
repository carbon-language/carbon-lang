
// CHECK-DAG: !DICompositeType(tag: DW_TAG_structure_type, name: "PR16214",{{.*}} line: [[@LINE+1]],{{.*}} isDefinition: true
struct PR16214 {
  int i;
};

typedef PR16214 bar;

bar *a;
bar b;

namespace PR14467 {
// CHECK-DAG: !DICompositeType(tag: DW_TAG_structure_type, name: "foo",{{.*}} line: [[@LINE+1]],{{.*}} isDefinition: true
struct foo {
};

foo *bar(foo *a) {
  foo *b = new foo(*a);
  return b;
}
}

namespace test1 {
// CHECK-DAG: !DICompositeType(tag: DW_TAG_structure_type, name: "foo",{{.*}} line: [[@LINE+1]],{{.*}} isDefinition: true
struct foo {
};

extern int bar(foo *a);
int baz(foo *a) {
  return bar(a);
}
}

namespace test2 {
// FIXME: if we were a bit fancier, we could realize that the 'foo' type is only
// required because of the 'bar' type which is not required at all (or might
// only be required to be declared)
// CHECK-DAG: !DICompositeType(tag: DW_TAG_structure_type, name: "foo",{{.*}} line: [[@LINE+1]],{{.*}} isDefinition: true
struct foo {
};

struct bar {
  foo f;
};

void func() {
  foo *f;
}
}
