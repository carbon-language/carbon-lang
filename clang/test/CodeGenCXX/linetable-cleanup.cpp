// RUN: %clang_cc1 -emit-llvm -g -triple x86_64-apple-darwin10 %s -o - | FileCheck %s

// Check the line numbers for cleanup code with EH in combination with
// simple return expressions.

// CHECK: define {{.*}}foo
// CHECK: call void @_ZN1CD1Ev(%class.C* {{.*}}), !dbg ![[CLEANUP:[0-9]+]]
// CHECK: ret i32 0, !dbg ![[RET:[0-9]+]]

// CHECK: define {{.*}}bar
// CHECK: ret void, !dbg ![[RETBAR:[0-9]+]]

// CHECK: define {{.*}}baz
// CHECK: ret void, !dbg ![[RETBAZ:[0-9]+]]

class C {
public:
  ~C() {}
  int i;
};

int foo()
{
  C c;
  c.i = 42;
  // This breakpoint should be at/before the cleanup code.
  // CHECK: ![[CLEANUP]] = !{i32 [[@LINE+1]], i32 0, !{{.*}}, null}
  return 0;
  // CHECK: ![[RET]] = !{i32 [[@LINE+1]], i32 0, !{{.*}}, null}
}

void bar()
{
  if (!foo())
    // CHECK: {{.*}} = !{i32 [[@LINE+1]], i32 0, !{{.*}}, null}
    return;

  if (foo()) {
    C c;
    c.i = foo();
  }
  // Clang creates only a single ret instruction. Make sure it is at a useful line.
  // CHECK: ![[RETBAR]] = !{i32 [[@LINE+1]], i32 0, !{{.*}}, null}
}

void baz()
{
  if (!foo())
    // CHECK: ![[SCOPE1:.*]] = !{!"0xb\00[[@LINE-1]]\00{{.*}}", {{.*}} ; [ DW_TAG_lexical_block ]
    // CHECK: {{.*}} = !{i32 [[@LINE+1]], i32 0, ![[SCOPE1]], null}
    return;

  if (foo()) {
    // no cleanup
    // CHECK: {{.*}} = !{i32 [[@LINE+2]], i32 0, ![[SCOPE2:.*]], null}
    // CHECK: ![[SCOPE2]] = !{!"0xb\00[[@LINE-3]]\00{{.*}}", {{.*}} ; [ DW_TAG_lexical_block ]
    return;
  }
  // CHECK: ![[RETBAZ]] = !{i32 [[@LINE+1]], i32 0, !{{.*}}, null}
}
