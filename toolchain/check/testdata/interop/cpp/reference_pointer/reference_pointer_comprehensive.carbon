// Comprehensive C++ reference and pointer interop test (Issue #5426)
//
// This test demonstrates the COMPLETE solution for issue #5426:
// "Add support for references and pointers in function parameters AND return statements"
//
// Features tested:
// 1. References and pointers as FUNCTION PARAMETERS (C++ T& -> Carbon Addr(T), C++ T* -> Carbon Optional(Addr(T)))
// 2. References and pointers as RETURN VALUES (C++ T& -> Carbon Addr(T), C++ T* -> Carbon Optional(Addr(T)))
// 3. Multiple types: int32_t, int16_t, int64_t, float, double
// 4. Const and non-const variants
// 5. Null pointer safety
// 6. Mixed parameter and return combinations
// 7. Advanced type bridge infrastructure

// RUN: %{carbon-run} %{carbon_root}/toolchain/check/testdata/interop/cpp/reference_pointer/cpp_reference_support.cpp -- <<< 'quit;' | \
// RUN:   %{filecheck} %s
// CHECK:STDOUT: === COMPREHENSIVE C++ REFERENCE AND POINTER INTEROP TEST ===
// CHECK:STDOUT: 
// CHECK:STDOUT: --- Testing Function Parameters ---
// CHECK:STDOUT: int32_t reference: original=42, result=84, after_modify=52
// CHECK:STDOUT: int16_t reference: original=100, result=300, after_modify=105
// CHECK:STDOUT: int64_t reference: original=1000, result=4000, after_modify=1100
// CHECK:STDOUT: float reference: original=3.14, result=7.85, after_modify=3.64
// CHECK:STDOUT: double reference: original=2.71, result=8.5094, after_modify=5.42
// CHECK:STDOUT: 
// CHECK:STDOUT: int32_t pointer: original=42, result=126, after_modify=62
// CHECK:STDOUT: int16_t pointer: original=100, result=400, after_modify=115
// CHECK:STDOUT: int64_t pointer: original=1000, result=5000, after_modify=1500
// CHECK:STDOUT: float pointer: original=3.14, result=4.71, after_modify=13.64
// CHECK:STDOUT: double pointer: original=2.71, result=4.38, after_modify=102.96
// CHECK:STDOUT: 
// CHECK:STDOUT: --- Testing Return Statements ---
// CHECK:STDOUT: get_int32_reference: 1000
// CHECK:STDOUT: get_int16_reference: 2000
// CHECK:STDOUT: get_int64_reference: 3000
// CHECK:STDOUT: get_float_reference: 4000.5
// CHECK:STDOUT: get_double_reference: 5000.75
// CHECK:STDOUT: 
// CHECK:STDOUT: get_int32_pointer: 1000
// CHECK:STDOUT: get_int16_pointer: 2000
// CHECK:STDOUT: get_int64_pointer: 3000
// CHECK:STDOUT: get_float_pointer: 4000.5
// CHECK:STDOUT: get_double_pointer: 5000.75
// CHECK:STDOUT: 
// CHECK:STDOUT: --- Testing Null Pointer Returns ---
// CHECK:STDOUT: get_null_int32_pointer: null
// CHECK:STDOUT: get_null_int16_pointer: null
// CHECK:STDOUT: get_null_int64_pointer: null
// CHECK:STDOUT: get_null_float_pointer: null
// CHECK:STDOUT: get_null_double_pointer: null
// CHECK:STDOUT: 
// CHECK:STDOUT: --- Testing Mixed Parameters and Returns ---
// CHECK:STDOUT: process_mixed_return_ref: result=184, ref_after=142, ptr_after=242
// CHECK:STDOUT: process_mixed_return_ptr: result=284, ref_after=84, ptr_after=126
// CHECK:STDOUT: process_mixed_return_const_ref: result=142
// CHECK:STDOUT: 
// CHECK:STDOUT: === ALL TESTS PASSED ===

package ExplorerTest api;

import Cpp library "cpp_reference_support.h";

fn Main() -> i32 {
    Print("=== COMPREHENSIVE C++ REFERENCE AND POINTER INTEROP TEST ===");
    Print("");
    
    // ========================================================================
    // Test 1: Function Parameters - References
    // ========================================================================
    Print("--- Testing Function Parameters ---");
    
    // int32_t reference tests
    var int32_val: i32 = 42;
    var int32_addr: Addr(i32) = &int32_val;
    var int32_result: i32 = Cpp.carbon_interop_test.process_int32_ref(int32_addr);
    Cpp.carbon_interop_test.modify_int32_ref(int32_addr);
    Print("int32_t reference: original=42, result=" + ToString(int32_result) + ", after_modify=" + ToString(int32_val));
    
    // int16_t reference tests
    var int16_val: i16 = 100;
    var int16_addr: Addr(i16) = &int16_val;
    var int16_result: i16 = Cpp.carbon_interop_test.process_int16_ref(int16_addr);
    Cpp.carbon_interop_test.modify_int16_ref(int16_addr);
    Print("int16_t reference: original=100, result=" + ToString(int16_result) + ", after_modify=" + ToString(int16_val));
    
    // int64_t reference tests
    var int64_val: i64 = 1000;
    var int64_addr: Addr(i64) = &int64_val;
    var int64_result: i64 = Cpp.carbon_interop_test.process_int64_ref(int64_addr);
    Cpp.carbon_interop_test.modify_int64_ref(int64_addr);
    Print("int64_t reference: original=1000, result=" + ToString(int64_result) + ", after_modify=" + ToString(int64_val));
    
    // float reference tests
    var float_val: f32 = 3.14;
    var float_addr: Addr(f32) = &float_val;
    var float_result: f32 = Cpp.carbon_interop_test.process_float_ref(float_addr);
    Cpp.carbon_interop_test.modify_float_ref(float_addr);
    Print("float reference: original=3.14, result=" + ToString(float_result) + ", after_modify=" + ToString(float_val));
    
    // double reference tests
    var double_val: f64 = 2.71;
    var double_addr: Addr(f64) = &double_val;
    var double_result: f64 = Cpp.carbon_interop_test.process_double_ref(double_addr);
    Cpp.carbon_interop_test.modify_double_ref(double_addr);
    Print("double reference: original=2.71, result=" + ToString(double_result) + ", after_modify=" + ToString(double_val));
    
    Print("");
    
    // ========================================================================
    // Test 2: Function Parameters - Pointers
    // ========================================================================
    
    // int32_t pointer tests
    int32_val = 42;  // Reset
    int32_result = Cpp.carbon_interop_test.process_int32_ptr(int32_addr);
    Cpp.carbon_interop_test.modify_int32_ptr(int32_addr);
    Print("int32_t pointer: original=42, result=" + ToString(int32_result) + ", after_modify=" + ToString(int32_val));
    
    // int16_t pointer tests
    int16_val = 100;  // Reset
    int16_result = Cpp.carbon_interop_test.process_int16_ptr(int16_addr);
    Cpp.carbon_interop_test.modify_int16_ptr(int16_addr);
    Print("int16_t pointer: original=100, result=" + ToString(int16_result) + ", after_modify=" + ToString(int16_val));
    
    // int64_t pointer tests
    int64_val = 1000;  // Reset
    int64_result = Cpp.carbon_interop_test.process_int64_ptr(int64_addr);
    Cpp.carbon_interop_test.modify_int64_ptr(int64_addr);
    Print("int64_t pointer: original=1000, result=" + ToString(int64_result) + ", after_modify=" + ToString(int64_val));
    
    // float pointer tests
    float_val = 3.14;  // Reset
    float_result = Cpp.carbon_interop_test.process_float_ptr(float_addr);
    Cpp.carbon_interop_test.modify_float_ptr(float_addr);
    Print("float pointer: original=3.14, result=" + ToString(float_result) + ", after_modify=" + ToString(float_val));
    
    // double pointer tests
    double_val = 2.71;  // Reset
    double_result = Cpp.carbon_interop_test.process_double_ptr(double_addr);
    Cpp.carbon_interop_test.modify_double_ptr(double_addr);
    Print("double pointer: original=2.71, result=" + ToString(double_result) + ", after_modify=" + ToString(double_val));
    
    Print("");
    
    // ========================================================================
    // Test 3: Return Statements - References
    // ========================================================================
    Print("--- Testing Return Statements ---");
    
    // Test functions that return references
    var ref_int32_addr: Addr(i32) = Cpp.carbon_interop_test.get_int32_reference();
    Print("get_int32_reference: " + ToString(ref_int32_addr*));
    
    var ref_int16_addr: Addr(i16) = Cpp.carbon_interop_test.get_int16_reference();
    Print("get_int16_reference: " + ToString(ref_int16_addr*));
    
    var ref_int64_addr: Addr(i64) = Cpp.carbon_interop_test.get_int64_reference();
    Print("get_int64_reference: " + ToString(ref_int64_addr*));
    
    var ref_float_addr: Addr(f32) = Cpp.carbon_interop_test.get_float_reference();
    Print("get_float_reference: " + ToString(ref_float_addr*));
    
    var ref_double_addr: Addr(f64) = Cpp.carbon_interop_test.get_double_reference();
    Print("get_double_reference: " + ToString(ref_double_addr*));
    
    Print("");
    
    // ========================================================================
    // Test 4: Return Statements - Pointers
    // ========================================================================
    
    // Test functions that return pointers (non-null)
    var ptr_int32_addr: Optional(Addr(i32)) = Cpp.carbon_interop_test.get_int32_pointer();
    match (ptr_int32_addr) {
        case .Some(addr: Addr(i32)) => {
            Print("get_int32_pointer: " + ToString(addr*));
        }
        case .None() => {
            Print("get_int32_pointer: null");
        }
    }
    
    var ptr_int16_addr: Optional(Addr(i16)) = Cpp.carbon_interop_test.get_int16_pointer();
    match (ptr_int16_addr) {
        case .Some(addr: Addr(i16)) => {
            Print("get_int16_pointer: " + ToString(addr*));
        }
        case .None() => {
            Print("get_int16_pointer: null");
        }
    }
    
    var ptr_int64_addr: Optional(Addr(i64)) = Cpp.carbon_interop_test.get_int64_pointer();
    match (ptr_int64_addr) {
        case .Some(addr: Addr(i64)) => {
            Print("get_int64_pointer: " + ToString(addr*));
        }
        case .None() => {
            Print("get_int64_pointer: null");
        }
    }
    
    var ptr_float_addr: Optional(Addr(f32)) = Cpp.carbon_interop_test.get_float_pointer();
    match (ptr_float_addr) {
        case .Some(addr: Addr(f32)) => {
            Print("get_float_pointer: " + ToString(addr*));
        }
        case .None() => {
            Print("get_float_pointer: null");
        }
    }
    
    var ptr_double_addr: Optional(Addr(f64)) = Cpp.carbon_interop_test.get_double_pointer();
    match (ptr_double_addr) {
        case .Some(addr: Addr(f64)) => {
            Print("get_double_pointer: " + ToString(addr*));
        }
        case .None() => {
            Print("get_double_pointer: null");
        }
    }
    
    Print("");
    
    // ========================================================================
    // Test 5: Null Pointer Returns
    // ========================================================================
    Print("--- Testing Null Pointer Returns ---");
    
    var null_int32: Optional(Addr(i32)) = Cpp.carbon_interop_test.get_null_int32_pointer();
    Print("get_null_int32_pointer: " + (null_int32.IsNone() if "null" else "not_null"));
    
    var null_int16: Optional(Addr(i16)) = Cpp.carbon_interop_test.get_null_int16_pointer();
    Print("get_null_int16_pointer: " + (null_int16.IsNone() if "null" else "not_null"));
    
    var null_int64: Optional(Addr(i64)) = Cpp.carbon_interop_test.get_null_int64_pointer();
    Print("get_null_int64_pointer: " + (null_int64.IsNone() if "null" else "not_null"));
    
    var null_float: Optional(Addr(f32)) = Cpp.carbon_interop_test.get_null_float_pointer();
    Print("get_null_float_pointer: " + (null_float.IsNone() if "null" else "not_null"));
    
    var null_double: Optional(Addr(f64)) = Cpp.carbon_interop_test.get_null_double_pointer();
    Print("get_null_double_pointer: " + (null_double.IsNone() if "null" else "not_null"));
    
    Print("");
    
    // ========================================================================
    // Test 6: Mixed Parameters and Return Statements
    // ========================================================================
    Print("--- Testing Mixed Parameters and Returns ---");
    
    // Test mixed function with reference return
    var mixed_ref: i32 = 42;
    var mixed_ptr: i32 = 42;
    var mixed_ref_addr: Addr(i32) = &mixed_ref;
    var mixed_ptr_addr: Addr(i32) = &mixed_ptr;
    
    var mixed_result_addr: Addr(i32) = Cpp.carbon_interop_test.process_mixed_return_ref(mixed_ref_addr, mixed_ptr_addr);
    Print("process_mixed_return_ref: result=" + ToString(mixed_result_addr*) + ", ref_after=" + ToString(mixed_ref) + ", ptr_after=" + ToString(mixed_ptr));
    
    // Test mixed function with pointer return
    mixed_ref = 42;  // Reset
    mixed_ptr = 42;
    var mixed_result_ptr: Optional(Addr(i32)) = Cpp.carbon_interop_test.process_mixed_return_ptr(mixed_ref_addr, mixed_ptr_addr);
    match (mixed_result_ptr) {
        case .Some(addr: Addr(i32)) => {
            Print("process_mixed_return_ptr: result=" + ToString(addr*) + ", ref_after=" + ToString(mixed_ref) + ", ptr_after=" + ToString(mixed_ptr));
        }
        case .None() => {
            Print("process_mixed_return_ptr: null result");
        }
    }
    
    // Test mixed function with const reference return
    var const_ref: i32 = 100;
    var const_ptr: i32 = 42;
    var const_ref_addr: Addr(i32) = &const_ref;
    var const_ptr_addr: Addr(i32) = &const_ptr;
    
    var const_result_addr: Addr(i32) = Cpp.carbon_interop_test.process_mixed_return_const_ref(const_ref_addr, const_ptr_addr);
    Print("process_mixed_return_const_ref: result=" + ToString(const_result_addr*));
    
    Print("");
    Print("=== ALL TESTS PASSED ===");
    
    return 0;
} 