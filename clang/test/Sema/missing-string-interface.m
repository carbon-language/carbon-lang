// RUN: clang %s -verify -fsyntax-only

@class NSString;

// GCC considers this an error, so clang will...
NSString *s = @"123"; // expected-error: {{cannot find interface declaration for 'NSConstantString'}}

