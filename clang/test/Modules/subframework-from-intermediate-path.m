// RUN: rm -rf %t
// RUN: %clang_cc1 -fmodules-cache-path=%t -fmodules -F %S/Inputs -F %S/Inputs/DependsOnModule.framework/Frameworks %s -verify

@import DependsOnModule;
@import SubFramework; // expected-error{{module 'SubFramework' not found}}
