// RUN: %clang_cc1 -triple r600-- -verify -fsyntax-only %s

typedef __attribute__((amdgpu_num_vgpr(128))) struct FooStruct { // expected-error {{'amdgpu_num_vgpr' attribute only applies to kernel functions}}
  int x;
  float y;
} FooStruct;


__attribute__((amdgpu_num_vgpr("ABC"))) kernel void foo2() {} // expected-error {{'amdgpu_num_vgpr' attribute requires an integer constant}}
__attribute__((amdgpu_num_sgpr("ABC"))) kernel void foo3() {} // expected-error {{'amdgpu_num_sgpr' attribute requires an integer constant}}


__attribute__((amdgpu_num_vgpr(40))) void foo4() {} // expected-error {{'amdgpu_num_vgpr' attribute only applies to kernel functions}}
__attribute__((amdgpu_num_sgpr(64))) void foo5() {} // expected-error {{'amdgpu_num_sgpr' attribute only applies to kernel functions}}

__attribute__((amdgpu_num_vgpr(40))) kernel void foo7() {}
__attribute__((amdgpu_num_sgpr(64))) kernel void foo8() {}
__attribute__((amdgpu_num_vgpr(40), amdgpu_num_sgpr(64))) kernel void foo9() {}

// Check 0 VGPR is accepted.
__attribute__((amdgpu_num_vgpr(0))) kernel void foo10() {}

// Check 0 SGPR is accepted.
__attribute__((amdgpu_num_sgpr(0))) kernel void foo11() {}

// Check both 0 SGPR and VGPR is accepted.
__attribute__((amdgpu_num_vgpr(0), amdgpu_num_sgpr(0))) kernel void foo12() {}

// Too large VGPR value.
__attribute__((amdgpu_num_vgpr(4294967296))) kernel void foo13() {} // expected-error {{integer constant expression evaluates to value 4294967296 that cannot be represented in a 32-bit unsigned integer type}}

__attribute__((amdgpu_num_sgpr(4294967296))) kernel void foo14() {} // expected-error {{integer constant expression evaluates to value 4294967296 that cannot be represented in a 32-bit unsigned integer type}}

__attribute__((amdgpu_num_sgpr(4294967296), amdgpu_num_vgpr(4294967296))) kernel void foo15() {} // expected-error 2 {{integer constant expression evaluates to value 4294967296 that cannot be represented in a 32-bit unsigned integer type}}


// Make sure it is accepted with kernel keyword before the attribute.
kernel __attribute__((amdgpu_num_vgpr(40))) void foo16() {}

kernel __attribute__((amdgpu_num_sgpr(40))) void foo17() {}
