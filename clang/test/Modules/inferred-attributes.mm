// RUN: rm -rf %t
// RUN: %clang_cc1 -fmodules -fmodules-cache-path=%t -F %S/Inputs/inferred-attr -fsyntax-only -verify %s
// expected-no-diagnostics
extern "C" {
@import InferredExternC;
}
