// RUN: %clang_cc1 -g -emit-llvm %s -o -| FileCheck %s
//
// Two variables with the same name in subsequent if staments need to be in separate scopes.
//
// rdar://problem/14024005

int src();

void f();

void func() {
  // CHECK: = metadata !{i32 786688, metadata [[IF1:![0-9]*]], {{.*}} ; [ DW_TAG_auto_variable ] [i] [line [[@LINE+2]]]
  // CHECK: [[IF1]] = metadata !{i32 {{.*}}, metadata !{{.*}}, i32 [[@LINE+1]], {{.*}}} ; [ DW_TAG_lexical_block ]
  if (int i = src())
    f();

  // CHECK: = metadata !{i32 786688, metadata [[IF2:![0-9]*]], {{.*}} ; [ DW_TAG_auto_variable ] [i] [line [[@LINE+2]]]
  // CHECK: [[IF2]] = metadata !{i32 {{.*}}, metadata !{{.*}}, i32 [[@LINE+1]], {{.*}}} ; [ DW_TAG_lexical_block ]
  if (int i = src()) {
    f();
  } else
    f();

  // CHECK: = metadata !{i32 786688, metadata [[FOR:![0-9]*]], {{.*}} ; [ DW_TAG_auto_variable ] [i] [line [[@LINE+2]]]
  // CHECK: [[FOR]] = metadata !{i32 {{.*}}, metadata !{{.*}}, i32 [[@LINE+1]], {{.*}}} ; [ DW_TAG_lexical_block ]
  for (int i = 0;
  // CHECK: = metadata !{i32 786688, metadata [[FOR_BODY:![0-9]*]], {{.*}} ; [ DW_TAG_auto_variable ] [b] [line [[@LINE+6]]]
  // The scope could be located at 'bool b', but LLVM drops line information for
  // scopes anyway, so it's not terribly important.
  // FIXME: change the debug info schema to not include locations of scopes,
  // since they're not used.
  // CHECK: [[FOR_BODY]] = metadata !{i32 {{.*}}, metadata [[FOR]], i32 [[@LINE-6]], {{.*}}} ; [ DW_TAG_lexical_block ]
       bool b = i != 10; ++i)
    f();

  // CHECK: = metadata !{i32 786688, metadata [[FOR:![0-9]*]], {{.*}} ; [ DW_TAG_auto_variable ] [i] [line [[@LINE+2]]]
  // CHECK: [[FOR]] = metadata !{i32 {{.*}}, metadata !{{.*}}, i32 [[@LINE+1]], {{.*}}} ; [ DW_TAG_lexical_block ]
  for (int i = 0; i != 10; ++i) {
  // FIXME: Do not include scopes that have only other scopes (and no variables
  // or using declarations) as direct children, they just waste
  // space/relocations/etc.
  // CHECK: [[FOR_BODY:![0-9]*]] = metadata !{i32 {{.*}}, metadata [[FOR]], i32 [[@LINE-4]], {{.*}}} ; [ DW_TAG_lexical_block ]
  // CHECK: = metadata !{i32 786688, metadata [[FOR_COMPOUND:![0-9]*]], {{.*}} ; [ DW_TAG_auto_variable ] [b] [line [[@LINE+2]]]
  // CHECK: [[FOR_COMPOUND]] = metadata !{i32 {{.*}}, metadata [[FOR_BODY]], i32 [[@LINE-6]], {{.*}}} ; [ DW_TAG_lexical_block ]
    bool b = i % 2;
  }
}
