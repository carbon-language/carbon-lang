// RUN: tblgen %s | FileCheck %s
// RUN: tblgen %s | FileCheck %s

class Base<int i> {
  int value = i;
}

multiclass Multi<list<int> values> {
  def ONE : Base<values[0]>;
  def TWO : Base<values[1]>;

  multidef COUNT<values, int v, 2> : Base<v>;
}

defm List : Multi<[1, 2, 3, 4, 5, 6]>;

// CHECK: MD4.ListCOUNT
// CHECK: int value = 6
