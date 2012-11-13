// RUN: %clang -emit-llvm -S -g %s -o - | FileCheck %s

// <rdar://problem/12566646>

// The subrange for a type 'int[1]' and 'int[0]' should be different. Use the
// 'count' attribute instead of the 'upper_bound' attribute do disabmiguate the
// DIE.

struct foo {
  int a;
  int b[1];
};

struct bar {
  int a;
  int b[0];
};

int main()
{
  struct foo my_foo;
  struct bar my_bar;

  my_foo.a = 3;
  my_bar.a = 5;

  return my_foo.a + my_bar.a;
}

// The third metadata operand is the count.
//
// CHECK: metadata !{i32 786465, i64 0, i64 0, i64 1} ; [ DW_TAG_subrange_type ]
// CHECK: metadata !{i32 786465, i64 0, i64 0, i64 0} ; [ DW_TAG_subrange_type ]
