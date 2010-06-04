// RUN: c-index-test -test-load-source local -fblocks %s | FileCheck %s

typedef int int_t;
struct foo { long x; };

void test() {
  static struct foo _foo;
  ^ int_t(struct foo *foo) { return (int_t) foo->x; }(&_foo);
}

// TODO: expose the BlockExpr, CastExpr, and UnaryOperatorExpr here

// CHECK: blocks.c:3:13: TypedefDecl=int_t:3:13 (Definition) Extent=[3:13 - 3:18]
// CHECK: blocks.c:4:8: StructDecl=foo:4:8 (Definition) Extent=[4:1 - 4:23]
// CHECK: blocks.c:4:19: FieldDecl=x:4:19 (Definition) Extent=[4:19 - 4:20]
// CHECK: blocks.c:6:6: FunctionDecl=test:6:6 (Definition) Extent=[6:6 - 9:2]
// CHECK: blocks.c:7:21: VarDecl=_foo:7:21 (Definition) Extent=[7:17 - 7:25]
// CHECK: blocks.c:7:17: TypeRef=struct foo:4:8 Extent=[7:17 - 7:20]
// CHECK: blocks.c:8:3: CallExpr= Extent=[8:3 - 8:61]
// CHECK: blocks.c:8:3: UnexposedExpr= Extent=[8:3 - 8:54]
// CHECK: blocks.c:8:5: TypeRef=int_t:3:13 Extent=[8:5 - 8:10]
// CHECK: blocks.c:8:23: ParmDecl=foo:8:23 (Definition) Extent=[8:18 - 8:26]
// CHECK: blocks.c:8:18: TypeRef=struct foo:4:8 Extent=[8:18 - 8:21]
// CHECK: blocks.c:8:37: UnexposedExpr=x:4:19 Extent=[8:37 - 8:51]
// CHECK: blocks.c:8:38: TypeRef=int_t:3:13 Extent=[8:38 - 8:43]
// CHECK: blocks.c:8:50: MemberRefExpr=x:4:19 Extent=[8:45 - 8:51]
// CHECK: blocks.c:8:45: DeclRefExpr=foo:8:23 Extent=[8:45 - 8:48]
// CHECK: blocks.c:8:55: UnexposedExpr= Extent=[8:55 - 8:60]
// CHECK: blocks.c:8:56: DeclRefExpr=_foo:7:21 Extent=[8:56 - 8:60]
