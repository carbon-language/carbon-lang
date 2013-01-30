// RUN: %clang_cc1 -fsyntax-only -verify %s
// RUN: %clang_cc1 -E -DPP_ONLY=1 %s -o %t
// RUN: FileCheck --strict-whitespace --input-file=%t %s

// This file contains Unicode characters; please do not "fix" them!

extern int x; // expected-warning {{treating Unicode character as whitespace}}
extern int　x; // expected-warning {{treating Unicode character as whitespace}}

// CHECK: extern int {{x}}
// CHECK: extern int　{{x}}

#if PP_ONLY
CHECK: The preprocessor should not complain about Unicode characters like ©.
#endif
