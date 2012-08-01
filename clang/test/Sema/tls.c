// Test that TLS is correctly considered supported or unsupported for the
// different targets.

// Linux supports TLS.
// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu %s
// RUN: %clang_cc1 -triple i386-pc-linux-gnu %s

// Darwin supports TLS since 10.7.
// RUN: %clang_cc1 -verify -triple x86_64-apple-darwin10 %s
// RUN: %clang_cc1 -triple x86_64-apple-macos10.7.0 %s

// FIXME: I thought it was supported actually?
// RUN: %clang_cc1 -verify -triple x86_64-pc-win32 %s
// RUN: %clang_cc1 -verify -triple i386-pc-win32 %s

// OpenBSD does not suppport TLS.
// RUN: %clang_cc1 -verify -triple x86_64-pc-openbsd %s
// RUN: %clang_cc1 -verify -triple i386-pc-openbsd %s

__thread int x; // expected-error {{thread-local storage is unsupported for the current target}}
