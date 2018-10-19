// RUN: %clang_cc1 -std=c++17 %s -emit-llvm -o - | FileCheck %s --implicit-check-not='call{{.*}}dtor'

void then();

struct dtor {
  ~dtor();
};

dtor ctor();

// [lifetime extension occurs if the object was obtained by]
//  -- a temporary materialization conversion
// CHECK-LABEL: ref_binding
void ref_binding() {
  // CHECK: call {{.*}}ctor
  auto &&x = ctor();
  // CHECK: call {{.*}}then
  then();
  // CHECK: call {{.*}}dtor
  // CHECK: }
}

//  -- ( expression )
// CHECK-LABEL: parens
void parens() {
  // CHECK: call {{.*}}ctor
  auto &&x = ctor();
  // CHECK: call {{.*}}then
  then();
  // CHECK: call {{.*}}dtor
  // CHECK: }
}

//  -- subscripting of an array
// CHECK-LABEL: array_subscript_1
void array_subscript_1() {
  using T = dtor[1];
  // CHECK: call {{.*}}ctor
  auto &&x = T{ctor()}[0];
  // CHECK: call {{.*}}then
  then();
  // CHECK: call {{.*}}dtor
  // CHECK: }
}
// CHECK-LABEL: array_subscript_2
void array_subscript_2() {
  using T = dtor[1];
  // CHECK: call {{.*}}ctor
  auto &&x = ((dtor*)T{ctor()})[0];
  // CHECK: call {{.*}}dtor
  // CHECK: call {{.*}}then
  then();
  // CHECK: }
}

struct with_member { dtor d; ~with_member(); };
struct with_ref_member { dtor &&d; ~with_ref_member(); };

//  -- a class member access using the . operator [...]
// CHECK-LABEL: member_access_1
void member_access_1() {
  // CHECK: call {{.*}}ctor
  auto &&x = with_member{ctor()}.d;
  // CHECK: call {{.*}}then
  then();
  // CHECK: call {{.*}}with_member
  // CHECK: }
}
// CHECK-LABEL: member_access_2
void member_access_2() {
  // CHECK: call {{.*}}ctor
  auto &&x = with_ref_member{ctor()}.d;
  // CHECK: call {{.*}}with_ref_member
  // CHECK: call {{.*}}dtor
  // CHECK: call {{.*}}then
  then();
  // CHECK: }
}
// CHECK-LABEL: member_access_3
void member_access_3() {
  // CHECK: call {{.*}}ctor
  auto &&x = (&(const with_member&)with_member{ctor()})->d;
  // CHECK: call {{.*}}with_member
  // CHECK: call {{.*}}then
  then();
  // CHECK: }
}

//  -- a pointer-to-member operation using the .* operator [...]
// CHECK-LABEL: member_ptr_access_1
void member_ptr_access_1() {
  // CHECK: call {{.*}}ctor
  auto &&x = with_member{ctor()}.*&with_member::d;
  // CHECK: call {{.*}}then
  then();
  // CHECK: call {{.*}}with_member
  // CHECK: }
}
// CHECK-LABEL: member_ptr_access_2
void member_ptr_access_2() {
  // CHECK: call {{.*}}ctor
  auto &&x = (&(const with_member&)with_member{ctor()})->*&with_member::d;
  // CHECK: call {{.*}}with_member
  // CHECK: call {{.*}}then
  then();
  // CHECK: }
}

//  -- a [named] cast [...]
// CHECK-LABEL: static_cast
void test_static_cast() {
  // CHECK: call {{.*}}ctor
  auto &&x = static_cast<dtor&&>(ctor());
  // CHECK: call {{.*}}then
  then();
  // CHECK: call {{.*}}dtor
  // CHECK: }
}
// CHECK-LABEL: const_cast
void test_const_cast() {
  // CHECK: call {{.*}}ctor
  auto &&x = const_cast<dtor&&>(ctor());
  // CHECK: call {{.*}}then
  then();
  // CHECK: call {{.*}}dtor
  // CHECK: }
}
// CHECK-LABEL: reinterpret_cast
void test_reinterpret_cast() {
  // CHECK: call {{.*}}ctor
  auto &&x = reinterpret_cast<dtor&&>(static_cast<dtor&&>(ctor()));
  // CHECK: call {{.*}}then
  then();
  // CHECK: call {{.*}}dtor
  // CHECK: }
}
// CHECK-LABEL: dynamic_cast
void test_dynamic_cast() {
  // CHECK: call {{.*}}ctor
  auto &&x = dynamic_cast<dtor&&>(ctor());
  // CHECK: call {{.*}}then
  then();
  // CHECK: call {{.*}}dtor
  // CHECK: }
}

//  -- [explicit cast notation is defined in terms of the above]
// CHECK-LABEL: c_style_cast
void c_style_cast() {
  // CHECK: call {{.*}}ctor
  auto &&x = (dtor&&)ctor();
  // CHECK: call {{.*}}then
  then();
  // CHECK: call {{.*}}dtor
  // CHECK: }
}
// CHECK-LABEL: function_style_cast
void function_style_cast() {
  // CHECK: call {{.*}}ctor
  using R = dtor&&;
  auto &&x = R(ctor());
  // CHECK: call {{.*}}then
  then();
  // CHECK: call {{.*}}dtor
  // CHECK: }
}

//  -- a conditional operator
// CHECK-LABEL: conditional
void conditional(bool b) {
  // CHECK: call {{.*}}ctor
  // CHECK: call {{.*}}ctor
  auto &&x = b ? (dtor&&)ctor() : (dtor&&)ctor();
  // CHECK: call {{.*}}then
  then();
  // CHECK: call {{.*}}dtor
  // CHECK: call {{.*}}dtor
  // CHECK: }
}

//  -- a comma expression
// CHECK-LABEL: comma
void comma() {
  // CHECK: call {{.*}}ctor
  auto &&x = (true, (dtor&&)ctor());
  // CHECK: call {{.*}}then
  then();
  // CHECK: call {{.*}}dtor
  // CHECK: }
}
