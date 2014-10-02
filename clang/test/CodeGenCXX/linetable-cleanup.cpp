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
  // CHECK: ![[CLEANUP]] = metadata !{i32 [[@LINE+1]], i32 0, metadata !{{.*}}, null}
  return 0;
  // CHECK: ![[RET]] = metadata !{i32 [[@LINE+1]], i32 0, metadata !{{.*}}, null}
}

void bar()
{
  if (!foo())
    // CHECK: {{.*}} = metadata !{i32 [[@LINE+1]], i32 0, metadata !{{.*}}, null}
    return;

  if (foo()) {
    C c;
    c.i = foo();
  }
  // Clang creates only a single ret instruction. Make sure it is at a useful line.
  // CHECK: ![[RETBAR]] = metadata !{i32 [[@LINE+1]], i32 0, metadata !{{.*}}, null}
}

void baz()
{
  if (!foo())
    // CHECK: ![[SCOPE1:.*]] = metadata !{{{.*}}, i32 [[@LINE-1]], {{.*}}} ; [ DW_TAG_lexical_block ]
    // CHECK: {{.*}} = metadata !{i32 [[@LINE+1]], i32 0, metadata ![[SCOPE1]], null}
    return;

  if (foo()) {
    // no cleanup
    // CHECK: {{.*}} = metadata !{i32 [[@LINE+2]], i32 0, metadata ![[SCOPE2:.*]], null}
    // CHECK: ![[SCOPE2]] = metadata !{{{.*}}, i32 [[@LINE-3]], {{.*}}} ; [ DW_TAG_lexical_block ]
    return;
  }
  // CHECK: ![[RETBAZ]] = metadata !{i32 [[@LINE+1]], i32 0, metadata !{{.*}}, null}
}
