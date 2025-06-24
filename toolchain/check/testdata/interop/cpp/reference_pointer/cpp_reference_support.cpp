#include "cpp_reference_support.h"

// C++ Reference and Pointer Support Implementation (Issue #5426)
// Complete implementation for references and pointers in parameters AND return statements

namespace carbon_interop_test {

// ============================================================================
// Global storage for return statement testing
// ============================================================================
int32_t global_int32 = 1000;
int16_t global_int16 = 2000;
int64_t global_int64 = 3000;
float global_float = 4000.5f;
double global_double = 5000.75;

// ============================================================================
// FUNCTION PARAMETERS - Reference parameter implementations
// ============================================================================

// int32_t reference implementations
int32_t process_int32_ref(int32_t& value) {
    return value * 2;
}

void modify_int32_ref(int32_t& value) {
    value += 10;
}

int32_t process_const_int32_ref(const int32_t& value) {
    return value + 1;
}

// int16_t reference implementations
int16_t process_int16_ref(int16_t& value) {
    return value * 3;
}

void modify_int16_ref(int16_t& value) {
    value += 5;
}

int16_t process_const_int16_ref(const int16_t& value) {
    return value + 2;
}

// int64_t reference implementations
int64_t process_int64_ref(int64_t& value) {
    return value * 4;
}

void modify_int64_ref(int64_t& value) {
    value += 100;
}

int64_t process_const_int64_ref(const int64_t& value) {
    return value + 10;
}

// float reference implementations
float process_float_ref(float& value) {
    return value * 2.5f;
}

void modify_float_ref(float& value) {
    value += 0.5f;
}

float process_const_float_ref(const float& value) {
    return value + 1.25f;
}

// double reference implementations
double process_double_ref(double& value) {
    return value * 3.14;
}

void modify_double_ref(double& value) {
    value += 2.71;
}

double process_const_double_ref(const double& value) {
    return value + 1.41;
}

// ============================================================================
// FUNCTION PARAMETERS - Pointer parameter implementations
// ============================================================================

// int32_t pointer implementations
int32_t process_int32_ptr(int32_t* value) {
    if (value == nullptr) {
        return -1; // Error value for null pointer
    }
    return *value * 3;
}

void modify_int32_ptr(int32_t* value) {
    if (value != nullptr) {
        *value += 20;
    }
}

int32_t process_const_int32_ptr(const int32_t* value) {
    if (value == nullptr) {
        return -1;
    }
    return *value + 5;
}

// int16_t pointer implementations
int16_t process_int16_ptr(int16_t* value) {
    if (value == nullptr) {
        return -1;
    }
    return *value * 4;
}

void modify_int16_ptr(int16_t* value) {
    if (value != nullptr) {
        *value += 15;
    }
}

int16_t process_const_int16_ptr(const int16_t* value) {
    if (value == nullptr) {
        return -1;
    }
    return *value + 7;
}

// int64_t pointer implementations
int64_t process_int64_ptr(int64_t* value) {
    if (value == nullptr) {
        return -1;
    }
    return *value * 5;
}

void modify_int64_ptr(int64_t* value) {
    if (value != nullptr) {
        *value += 500;
    }
}

int64_t process_const_int64_ptr(const int64_t* value) {
    if (value == nullptr) {
        return -1;
    }
    return *value + 25;
}

// float pointer implementations
float process_float_ptr(float* value) {
    if (value == nullptr) {
        return -1.0f;
    }
    return *value * 1.5f;
}

void modify_float_ptr(float* value) {
    if (value != nullptr) {
        *value += 10.5f;
    }
}

float process_const_float_ptr(const float* value) {
    if (value == nullptr) {
        return -1.0f;
    }
    return *value + 2.5f;
}

// double pointer implementations
double process_double_ptr(double* value) {
    if (value == nullptr) {
        return -1.0;
    }
    return *value * 1.618;
}

void modify_double_ptr(double* value) {
    if (value != nullptr) {
        *value += 100.25;
    }
}

double process_const_double_ptr(const double* value) {
    if (value == nullptr) {
        return -1.0;
    }
    return *value + 50.75;
}

// ============================================================================
// RETURN STATEMENTS - Functions that RETURN references and pointers
// ============================================================================

// Functions that return references (C++ T& -> Carbon Addr(T))
int32_t& get_int32_reference() {
    return global_int32;
}

const int32_t& get_const_int32_reference() {
    return global_int32;
}

int16_t& get_int16_reference() {
    return global_int16;
}

const int16_t& get_const_int16_reference() {
    return global_int16;
}

int64_t& get_int64_reference() {
    return global_int64;
}

const int64_t& get_const_int64_reference() {
    return global_int64;
}

float& get_float_reference() {
    return global_float;
}

const float& get_const_float_reference() {
    return global_float;
}

double& get_double_reference() {
    return global_double;
}

const double& get_const_double_reference() {
    return global_double;
}

// Functions that return pointers (C++ T* -> Carbon Optional(Addr(T)))
int32_t* get_int32_pointer() {
    return &global_int32;
}

const int32_t* get_const_int32_pointer() {
    return &global_int32;
}

int32_t* get_null_int32_pointer() {
    return nullptr;  // Test null pointer return
}

int16_t* get_int16_pointer() {
    return &global_int16;
}

const int16_t* get_const_int16_pointer() {
    return &global_int16;
}

int16_t* get_null_int16_pointer() {
    return nullptr;
}

int64_t* get_int64_pointer() {
    return &global_int64;
}

const int64_t* get_const_int64_pointer() {
    return &global_int64;
}

int64_t* get_null_int64_pointer() {
    return nullptr;
}

float* get_float_pointer() {
    return &global_float;
}

const float* get_const_float_pointer() {
    return &global_float;
}

float* get_null_float_pointer() {
    return nullptr;
}

double* get_double_pointer() {
    return &global_double;
}

const double* get_const_double_pointer() {
    return &global_double;
}

double* get_null_double_pointer() {
    return nullptr;
}

// ============================================================================
// MIXED PARAMETERS AND RETURN STATEMENTS
// ============================================================================

// Functions with mixed reference and pointer parameters AND return values
int32_t& process_mixed_return_ref(int32_t& ref_param, int32_t* ptr_param) {
    // Process both reference and pointer, modify global, return reference to global
    global_int32 = ref_param * 2;
    if (ptr_param != nullptr) {
        global_int32 += *ptr_param;
    }
    
    // Modify input parameters
    ref_param += 100;
    if (ptr_param != nullptr) {
        *ptr_param += 200;
    }
    
    return global_int32;
}

int32_t* process_mixed_return_ptr(int32_t& ref_param, int32_t* ptr_param) {
    // Process inputs and potentially return pointer to global or nullptr
    if (ptr_param == nullptr) {
        return nullptr;  // Return null if input pointer is null
    }
    
    global_int32 = ref_param + *ptr_param;
    ref_param *= 2;
    *ptr_param *= 3;
    
    return &global_int32;
}

const int32_t& process_mixed_return_const_ref(const int32_t& ref_param, const int32_t* ptr_param) {
    // Process const inputs, update global, return const reference
    global_int32 = ref_param;
    if (ptr_param != nullptr) {
        global_int32 += *ptr_param;
    }
    
    return global_int32;
}

// ============================================================================
// ADVANCED TYPE BRIDGE FUNCTIONS
// ============================================================================

// Functions using the CarbonAddr/CarbonPtr infrastructure
CarbonAddr<int32_t> create_carbon_addr(int32_t value) {
    return CarbonAddr<int32_t>::create_owned(value);
}

CarbonPtr<int32_t> create_carbon_ptr(int32_t* value) {
    return CarbonPtr<int32_t>(value);
}

CarbonPtr<int32_t> create_null_carbon_ptr() {
    return CarbonPtr<int32_t>();  // Creates null pointer
}

} // namespace carbon_interop_test 