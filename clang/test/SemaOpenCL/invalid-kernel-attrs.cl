// RUN: %clang_cc1 -verify %s 

kernel __attribute__((vec_type_hint)) void kernel1() {} //expected-error{{attribute takes one argument}}

kernel __attribute__((vec_type_hint(not_type))) void kernel2() {} //expected-error{{unknown type name 'not_type'}}

kernel __attribute__((vec_type_hint(void))) void kernel3() {} //expected-error{{invalid attribute argument 'void' - expecting a vector or vectorizable scalar type}}

kernel __attribute__((vec_type_hint(bool))) void kernel4() {} //expected-error{{invalid attribute argument 'bool' - expecting a vector or vectorizable scalar type}}

kernel __attribute__((vec_type_hint(int))) __attribute__((vec_type_hint(float))) void kernel5() {} //expected-warning{{attribute 'vec_type_hint' is already applied with different parameters}}

kernel __attribute__((work_group_size_hint(8,16,32,4))) void kernel6() {} //expected-error{{attribute requires exactly 3 arguments}}

kernel __attribute__((work_group_size_hint(1,2,3))) __attribute__((work_group_size_hint(3,2,1))) void kernel7() {}  //expected-warning{{attribute 'work_group_size_hint' is already applied with different parameters}}

