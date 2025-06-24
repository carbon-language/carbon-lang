#pragma once

// C++ Reference and Pointer Support for Carbon Interop (Issue #5426)
// This header demonstrates the core problem that issue #5426 aims to solve:
// Carbon needs to be able to import and call C++ functions with reference and pointer parameters

#include <cstdint>

namespace carbon_interop_test {

// Basic reference parameter functions that Carbon should be able to call
// These map C++ T& -> Carbon Addr(T)
int32_t process_int_ref(int32_t& value);
void modify_int_ref(int32_t& value);
int32_t process_const_int_ref(const int32_t& value);

// Basic pointer parameter functions that Carbon should be able to call  
// These map C++ T* -> Carbon Optional(Addr(T))
int32_t process_int_ptr(int32_t* value);
void modify_int_ptr(int32_t* value);
int32_t process_const_int_ptr(const int32_t* value);

// Mixed reference and pointer parameters
int32_t process_mixed(int32_t& ref_param, int32_t* ptr_param);

} // namespace carbon_interop_test 