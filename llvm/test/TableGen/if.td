// RUN: tblgen %s | FileCheck %s
// XFAIL: vg_leak

// Support for an `!if' operator as part of a `let' statement.
// CHECK:      class C
// CHECK-NEXT: bits<16> n = { ?, ?, ?, ?, ?, ?, ?, !if({ C:x{2} }, 0, 1), !if({ C:x{2} }, 1, 1), !if({ C:x{2} }, 0, 0), !if({ C:x{1} }, C:y{3}, 0), !if({ C:x{1} }, C:y{2}, 1), !if({ C:x{0} }, C:y{3}, C:z), !if({ C:x{0} }, C:y{2}, C:y{2}), !if({ C:x{0} }, C:y{1}, C:y{1}), !if({ C:x{0} }, C:y{0}, C:y{0}) };
class C<bits<3> x, bits<4> y, bit z> {
  bits<16> n;

  let n{8-6} = !if(x{2}, 0b010, 0b110);
  let n{5-4} = !if(x{1}, y{3-2}, {0, 1});
  let n{3-0} = !if(x{0}, y{3-0}, {z, y{2}, y{1}, y{0}});
}

// CHECK:      def One
// CHECK-NEXT: list<int> first = [1, 2, 3];
// CHECK-NEXT: list<int> rest = [1, 2, 3];

// CHECK:      def OneB
// CHECK-NEXT: list<int> vals = [1, 2, 3];

// CHECK:      def Two
// CHECK-NEXT: list<int> first = [1, 2, 3];
// CHECK-NEXT: list<int> rest = [4, 5, 6];

// CHECK:      def TwoB
// CHECK-NEXT: list<int> vals = [4, 5, 6];

class A<list<list<int>> vals> {
  list<int> first = vals[0];
  list<int> rest  = !if(!null(!cdr(vals)), vals[0], vals[1]);
}

def One : A<[[1,2,3]]>;
def Two : A<[[1,2,3], [4,5,6]]>;

class B<list<int> v> {
  list<int> vals = v;
}

class BB<list<list<int>> vals> : B<!if(!null(!cdr(vals)), vals[0], vals[1])>;
class BBB<list<list<int>> vals> : BB<vals>;

def OneB : BBB<[[1,2,3]]>;
def TwoB : BBB<[[1,2,3],[4,5,6]]>;
