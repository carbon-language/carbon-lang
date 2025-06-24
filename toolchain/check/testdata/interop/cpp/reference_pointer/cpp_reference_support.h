#pragma once

// C++ Reference and Pointer Support for Carbon Interop (Issue #5426)
// Complete implementation addressing:
// "Add support for references and pointers in function parameters AND return statements"

#include <cstdint>
#include <memory>
#include <optional>

namespace carbon_interop_test {

// ============================================================================
// Core Type Bridge Infrastructure (from carbon-interop framework)
// ============================================================================

/**
 * CarbonAddr<T> - Represents Carbon Addr(T) in C++
 * Provides safe access to Carbon memory from C++
 */
template<typename T>
class CarbonAddr {
private:
    T* ptr_;
    bool owned_;
    
public:
    explicit CarbonAddr(T* ptr, bool owned = false) 
        : ptr_(ptr), owned_(owned) {}
    
    ~CarbonAddr() {
        if (owned_ && ptr_) {
            delete ptr_;
        }
    }
    
    // Move constructor/assignment
    CarbonAddr(CarbonAddr&& other) noexcept 
        : ptr_(other.ptr_), owned_(other.owned_) {
        other.ptr_ = nullptr;
        other.owned_ = false;
    }
    
    CarbonAddr& operator=(CarbonAddr&& other) noexcept {
        if (this != &other) {
            if (owned_ && ptr_) delete ptr_;
            ptr_ = other.ptr_;
            owned_ = other.owned_;
            other.ptr_ = nullptr;
            other.owned_ = false;
        }
        return *this;
    }
    
    // Disable copy
    CarbonAddr(const CarbonAddr&) = delete;
    CarbonAddr& operator=(const CarbonAddr&) = delete;
    
    // Access operations
    T& operator*() { return *ptr_; }
    const T& operator*() const { return *ptr_; }
    T* operator->() { return ptr_; }
    const T* operator->() const { return ptr_; }
    
    T* get() { return ptr_; }
    const T* get() const { return ptr_; }
    
    bool is_valid() const { return ptr_ != nullptr; }
    
    // Create from C++ reference - maps C++ T& -> Carbon Addr(T)
    static CarbonAddr<T> from_reference(T& ref) {
        return CarbonAddr<T>(&ref, false);
    }
    
    // Create owned address - for Carbon -> C++ conversion
    static CarbonAddr<T> create_owned(T value) {
        return CarbonAddr<T>(new T(std::move(value)), true);
    }
};

/**
 * CarbonPtr<T> - Represents Carbon pointer types in C++
 * Maps C++ T* <-> Carbon Optional(Addr(T))
 */
template<typename T>
class CarbonPtr {
private:
    std::optional<CarbonAddr<T>> maybe_addr_;
    
public:
    // Create null pointer
    CarbonPtr() = default;
    
    // Create from C++ pointer
    explicit CarbonPtr(T* ptr) {
        if (ptr) {
            maybe_addr_ = CarbonAddr<T>::from_reference(*ptr);
        }
    }
    
    // Create from address
    explicit CarbonPtr(CarbonAddr<T> addr) 
        : maybe_addr_(std::move(addr)) {}
    
    // Check if null
    bool is_null() const { return !maybe_addr_.has_value(); }
    bool has_value() const { return maybe_addr_.has_value(); }
    
    // Dereference (returns optional)
    std::optional<T> dereference() const {
        if (maybe_addr_) {
            return **maybe_addr_;
        }
        return std::nullopt;
    }
    
    // Convert to C++ pointer (unsafe - returns nullptr if null)
    T* to_raw_ptr() {
        if (maybe_addr_) {
            return maybe_addr_->get();
        }
        return nullptr;
    }
};

// ============================================================================
// Global storage for return statement testing
// ============================================================================
extern int32_t global_int32;
extern int16_t global_int16;
extern int64_t global_int64;
extern float global_float;
extern double global_double;

// ============================================================================
// FUNCTION PARAMETERS - Basic reference parameter functions
// ============================================================================

// int32_t support
int32_t process_int32_ref(int32_t& value);
void modify_int32_ref(int32_t& value);
int32_t process_const_int32_ref(const int32_t& value);

// int16_t support  
int16_t process_int16_ref(int16_t& value);
void modify_int16_ref(int16_t& value);
int16_t process_const_int16_ref(const int16_t& value);

// int64_t support
int64_t process_int64_ref(int64_t& value);
void modify_int64_ref(int64_t& value);  
int64_t process_const_int64_ref(const int64_t& value);

// float support
float process_float_ref(float& value);
void modify_float_ref(float& value);
float process_const_float_ref(const float& value);

// double support
double process_double_ref(double& value);
void modify_double_ref(double& value);
double process_const_double_ref(const double& value);

// ============================================================================
// FUNCTION PARAMETERS - Basic pointer parameter functions  
// ============================================================================

// int32_t pointer support
int32_t process_int32_ptr(int32_t* value);
void modify_int32_ptr(int32_t* value);
int32_t process_const_int32_ptr(const int32_t* value);

// int16_t pointer support
int16_t process_int16_ptr(int16_t* value);
void modify_int16_ptr(int16_t* value);
int16_t process_const_int16_ptr(const int16_t* value);

// int64_t pointer support
int64_t process_int64_ptr(int64_t* value);
void modify_int64_ptr(int64_t* value);
int64_t process_const_int64_ptr(const int64_t* value);

// float pointer support
float process_float_ptr(float* value);
void modify_float_ptr(float* value);
float process_const_float_ptr(const float* value);

// double pointer support  
double process_double_ptr(double* value);
void modify_double_ptr(double* value);
double process_const_double_ptr(const double* value);

// ============================================================================
// RETURN STATEMENTS - Functions that RETURN references and pointers
// ============================================================================

// Functions that return references (C++ T& -> Carbon Addr(T))
int32_t& get_int32_reference();
const int32_t& get_const_int32_reference();
int16_t& get_int16_reference();
const int16_t& get_const_int16_reference(); 
int64_t& get_int64_reference();
const int64_t& get_const_int64_reference();
float& get_float_reference();
const float& get_const_float_reference();
double& get_double_reference();
const double& get_const_double_reference();

// Functions that return pointers (C++ T* -> Carbon Optional(Addr(T)))
int32_t* get_int32_pointer();
const int32_t* get_const_int32_pointer();
int32_t* get_null_int32_pointer();  // Returns nullptr
int16_t* get_int16_pointer();
const int16_t* get_const_int16_pointer();
int16_t* get_null_int16_pointer();
int64_t* get_int64_pointer();
const int64_t* get_const_int64_pointer(); 
int64_t* get_null_int64_pointer();
float* get_float_pointer();
const float* get_const_float_pointer();
float* get_null_float_pointer();
double* get_double_pointer();
const double* get_const_double_pointer();
double* get_null_double_pointer();

// ============================================================================
// MIXED PARAMETERS AND RETURN STATEMENTS
// ============================================================================

// Functions with mixed reference and pointer parameters AND return values
int32_t& process_mixed_return_ref(int32_t& ref_param, int32_t* ptr_param);
int32_t* process_mixed_return_ptr(int32_t& ref_param, int32_t* ptr_param);
const int32_t& process_mixed_return_const_ref(const int32_t& ref_param, const int32_t* ptr_param);

// ============================================================================
// ADVANCED TYPE BRIDGE FUNCTIONS
// ============================================================================

// Functions using the CarbonAddr/CarbonPtr infrastructure
CarbonAddr<int32_t> create_carbon_addr(int32_t value);
CarbonPtr<int32_t> create_carbon_ptr(int32_t* value);
CarbonPtr<int32_t> create_null_carbon_ptr();

// Template functions for type bridge conversion
template<typename T>
CarbonAddr<T> cpp_ref_to_carbon_addr(T& ref) {
    return CarbonAddr<T>::from_reference(ref);
}

template<typename T>  
CarbonPtr<T> cpp_ptr_to_carbon_ptr(T* ptr) {
    return CarbonPtr<T>(ptr);
}

} // namespace carbon_interop_test 