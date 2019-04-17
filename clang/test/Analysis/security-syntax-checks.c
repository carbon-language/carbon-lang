// RUN: %clang_analyze_cc1 %s -verify \
// RUN:   -analyzer-checker=security.insecureAPI

void builtin_function_call_crash_fixes(char *c) {
  __builtin_strncpy(c, "", 6); // expected-warning{{Call to function 'strncpy' is insecure as it does not provide security checks introduced in the C11 standard.}}
  __builtin_memset(c, '\0', (0)); // expected-warning{{Call to function 'memset' is insecure as it does not provide security checks introduced in the C11 standard.}}
  __builtin_memcpy(c, c, 0); // expected-warning{{Call to function 'memcpy' is insecure as it does not provide security checks introduced in the C11 standard.}}
}
