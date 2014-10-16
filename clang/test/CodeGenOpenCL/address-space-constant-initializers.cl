// RUN: %clang_cc1 %s -ffake-address-space-map -emit-llvm -o - | FileCheck %s

typedef struct {
    int i;
    float f; // At non-zero offset.
} ArrayStruct;

__constant ArrayStruct constant_array_struct = { 0, 0.0f };

typedef struct {
    __constant float* constant_float_ptr;
} ConstantArrayPointerStruct;

// CHECK: %struct.ConstantArrayPointerStruct = type { float addrspace(3)* }
// CHECK: addrspace(3) constant %struct.ConstantArrayPointerStruct { float addrspace(3)* bitcast (i8 addrspace(3)* getelementptr (i8 addrspace(3)* bitcast (%struct.ArrayStruct addrspace(3)* @constant_array_struct to i8 addrspace(3)*), i64 4) to float addrspace(3)*) }
// Bug  18567
__constant ConstantArrayPointerStruct constant_array_pointer_struct = {
    &constant_array_struct.f
};

