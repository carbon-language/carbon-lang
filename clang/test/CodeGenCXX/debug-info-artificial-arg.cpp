// RUN: %clang_cc1 -emit-llvm -g -triple x86_64-apple-darwin %s -o - | FileCheck %s

template<class X> class B {
public:
  explicit B(X* p = 0);
};

class A
{
public:
  A(int value) : m_a_value(value) {};
  A(int value, A* client_A) : m_a_value (value), m_client_A (client_A) {}

  virtual ~A() {}

private:
  int m_a_value;
  B<A> m_client_A;
};

int main(int argc, char **argv) {
  A reallyA (500);
}

// FIXME: The numbers are truly awful.
// CHECK: !18 = metadata !{i32 {{.*}}, i32 0, metadata !"", i32 0, i32 0, i64 64, i64 64, i64 0, i32 64, metadata !19} ; [ DW_TAG_pointer_type ]
// CHECK: !19 = metadata !{i32 {{.*}}, null, metadata !"A", metadata !6, i32 8, i64 128, i64 64, i32 0, i32 0, null, metadata !20, i32 0, metadata !19, null} ; [ DW_TAG_class_type ]
// CHECK: metadata !19, metadata !"A", metadata !"A", metadata !"", metadata !6, i32 12, metadata !45, i1 false, i1 false, i32 0, i32 0, null, i32 256, i1 false, null, null, i32 0, metadata !47, i32 12} ; [ DW_TAG_subprogram ]
// CHECK: metadata !"", i32 0, i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !46, i32 0, i32 0} ; [ DW_TAG_subroutine_type ]
// CHECK: !46 = metadata !{null, metadata !18, metadata !9, metadata !34}
