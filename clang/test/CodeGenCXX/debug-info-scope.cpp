// RUN: %clang_cc1 -g -std=c++11 -emit-llvm %s -o -| FileCheck %s
//
// Two variables with the same name in subsequent if staments need to be in separate scopes.
//
// rdar://problem/14024005

int src();

void f();

void func() {
  // CHECK: = !{!"0x100\00{{.*}}", [[IF1:![0-9]*]], {{.*}} ; [ DW_TAG_auto_variable ] [i] [line [[@LINE+2]]]
  // CHECK: [[IF1]] = !{!"0xb\00[[@LINE+1]]\00{{.*}}", !{{.*}}} ; [ DW_TAG_lexical_block ]
  if (int i = src())
    f();

  // CHECK: = !{!"0x100\00{{.*}}", [[IF2:![0-9]*]], {{.*}} ; [ DW_TAG_auto_variable ] [i] [line [[@LINE+2]]]
  // CHECK: [[IF2]] = !{!"0xb\00[[@LINE+1]]\00{{.*}}", !{{.*}}} ; [ DW_TAG_lexical_block ]
  if (int i = src()) {
    f();
  } else
    f();

  // CHECK: = !{!"0x100\00{{.*}}", [[FOR:![0-9]*]], {{.*}} ; [ DW_TAG_auto_variable ] [i] [line [[@LINE+2]]]
  // CHECK: [[FOR]] = !{!"0xb\00[[@LINE+1]]\00{{.*}}", !{{.*}}} ; [ DW_TAG_lexical_block ]
  for (int i = 0;
  // CHECK: = !{!"0x100\00{{.*}}", [[FOR_BODY:![0-9]*]], {{.*}} ; [ DW_TAG_auto_variable ] [b] [line [[@LINE+6]]]
  // The scope could be located at 'bool b', but LLVM drops line information for
  // scopes anyway, so it's not terribly important.
  // FIXME: change the debug info schema to not include locations of scopes,
  // since they're not used.
  // CHECK: [[FOR_BODY]] = !{!"0xb\00[[@LINE-6]]\00{{.*}}", !{{.*}}} ; [ DW_TAG_lexical_block ]
       bool b = i != 10; ++i)
    f();

  // CHECK: = !{!"0x100\00{{.*}}", [[FOR:![0-9]*]], {{.*}} ; [ DW_TAG_auto_variable ] [i] [line [[@LINE+2]]]
  // CHECK: [[FOR]] = !{!"0xb\00[[@LINE+1]]\00{{.*}}", !{{.*}}} ; [ DW_TAG_lexical_block ]
  for (int i = 0; i != 10; ++i) {
    // FIXME: Do not include scopes that have only other scopes (and no variables
    // or using declarations) as direct children, they just waste
    // space/relocations/etc.
    // CHECK: [[FOR_LOOP_INCLUDING_COND:!.*]] = !{!"0xb\00[[@LINE-4]]\00{{.*}}", !{{[0-9]+}}, [[FOR]]} ; [ DW_TAG_lexical_block ]
    // CHECK: = !{!"0x100\00{{.*}}", [[FOR_COMPOUND:![0-9]*]], {{.*}} ; [ DW_TAG_auto_variable ] [b] [line [[@LINE+2]]]
    // CHECK: [[FOR_COMPOUND]] = !{!"0xb\00[[@LINE-6]]\00{{.*}}", !{{[0-9]+}}, [[FOR_LOOP_INCLUDING_COND]]} ; [ DW_TAG_lexical_block ]
    bool b = i % 2;
  }

  int x[] = {1, 2};
  // CHECK: = !{!"0x100\00{{.*}}", [[RANGE_FOR:![0-9]*]], {{.*}} ; [ DW_TAG_auto_variable ] [__range] [line 0]
  // CHECK: [[RANGE_FOR]] = !{!"0xb\00[[@LINE+1]]\00{{.*}}", !{{.*}}} ; [ DW_TAG_lexical_block ]
  for (int i : x) {
  // CHECK: = !{!"0x100\00{{.*}}", [[RANGE_FOR_BODY:![0-9]*]], {{.*}} ; [ DW_TAG_auto_variable ] [i] [line [[@LINE-1]]]
  // CHECK: [[RANGE_FOR_BODY]] = !{!"0xb\00[[@LINE-2]]\00{{.*}}", !{{[0-9]+}}, [[RANGE_FOR]]} ; [ DW_TAG_lexical_block ]
  }
}
