// RUN: rm -rf %t
// RUN: %clang_cc1 -x objective-c -Wauto-import -fmodules-cache-path=%t -fmodules -fimplicit-module-maps -F %S/Inputs %s -verify

#include <NotAModule/NotAModule.h>

@import NotAModule; // expected-error{{module 'NotAModule' not found}}


