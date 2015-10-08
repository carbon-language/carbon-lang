// RUN: %clang_cc1 -triple x86_64-unk-unk -debug-info-kind=standalone -o - -emit-llvm %s | FileCheck %s
// On Darwin, "full" debug info is the default, so really these tests are
// identical, as cc1 no longer chooses the effective value of DebugInfoKind.
// RUN: %clang_cc1 -triple x86_64-apple-darwin -debug-info-kind=standalone -o - -emit-llvm %s | FileCheck %s

namespace rdar14101097_1 { // see also PR16214
// Check that we emit debug info for the definition of a struct if the
// definition is available, even if it's used via a pointer wrapped in a
// typedef.
// CHECK: !DICompositeType(tag: DW_TAG_structure_type, name: "foo"
// CHECK-NOT:              DIFlagFwdDecl
// CHECK-SAME:             ){{$}}
struct foo {
};

typedef foo *foop;

void bar() {
  foop f;
}
}

namespace rdar14101097_2 {
// As above, except trickier because we first encounter only a declaration of
// the type and no debug-info related use after we see the definition of the
// type.
// CHECK: !DICompositeType(tag: DW_TAG_structure_type, name: "foo"
// CHECK-NOT:              DIFlagFwdDecl
// CHECK-SAME:             ){{$}}
struct foo;
void bar() {
  foo *f;
}
struct foo {
};
}
