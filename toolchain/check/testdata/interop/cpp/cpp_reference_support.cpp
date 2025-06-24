#include "cpp_reference_support.h"

// C++ Reference and Pointer Support Implementation (Issue #5426)
// This file provides actual working C++ implementations that Carbon can call

namespace carbon_interop_test {

// Reference parameter implementations
int32_t process_int_ref(int32_t& value) {
    // Process the reference and return doubled value
    return value * 2;
}

void modify_int_ref(int32_t& value) {
    // Modify the value through reference
    value += 10;
}

int32_t process_const_int_ref(const int32_t& value) {
    // Process const reference, return incremented value
    return value + 1;
}

// Pointer parameter implementations  
int32_t process_int_ptr(int32_t* value) {
    // Check for null pointer and process
    if (value == nullptr) {
        return -1; // Error value for null pointer
    }
    return *value * 3;
}

void modify_int_ptr(int32_t* value) {
    // Modify through pointer if not null
    if (value != nullptr) {
        *value += 20;
    }
}

int32_t process_const_int_ptr(const int32_t* value) {
    // Process const pointer
    if (value == nullptr) {
        return -1;
    }
    return *value + 5;
}

// Mixed reference and pointer parameters
int32_t process_mixed(int32_t& ref_param, int32_t* ptr_param) {
    // Process both reference and pointer
    int32_t ref_result = ref_param * 2;
    int32_t ptr_result = (ptr_param != nullptr) ? (*ptr_param * 3) : 0;
    
    // Modify both
    ref_param += 100;
    if (ptr_param != nullptr) {
        *ptr_param += 200;
    }
    
    return ref_result + ptr_result;
}

} // namespace carbon_interop_test 