// RUN: %clang_cc1 -triple x86_64-unk-unk -fno-limit-debug-info -o - -emit-llvm -g %s | FileCheck %s
// Check that we emit debug info for a struct even if it is typedef'd before using.
// rdar://problem/14101097
//
// FIXME: This should work with -flimit-debug-info, too.

// Make sure this is not a forward declaration.
// CHECK-NOT: [ DW_TAG_structure_type ] [elusive_s] {{.*}} [fwd]
// CHECK: [ DW_TAG_member ] [foo]
// CHECK: [ DW_TAG_member ] [bar]
struct elusive_s {
  int foo;
  float bar;
};
typedef struct elusive_s* elusive_t;

int baz(void* x) {
  elusive_t s = x;
  return s->foo;
}
