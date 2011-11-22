// RUN: %clang_cc1 -E -faddress-sanitizer %s -o - | FileCheck --check-prefix=CHECK-ASAN %s
// RUN: %clang_cc1 -E  %s -o - | FileCheck --check-prefix=CHECK-NO-ASAN %s

#if __has_feature(address_sanitizer)
int AddressSanitizerEnabled();
#else
int AddressSanitizerDisabled();
#endif

// CHECK-ASAN: AddressSanitizerEnabled
// CHECK-NO-ASAN: AddressSanitizerDisabled
