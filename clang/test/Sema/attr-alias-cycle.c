// RUN: %clang_cc1 -triple x86_64-pc-linux  -fsyntax-only -verify -emit-llvm-only %s

// FIXME: The attributes use mangled names. Since we only keep a mapping from
// mangled name to llvm GlobalValue, we don't see the clang level decl for
// an alias target when constructing the alias. Given that and that alias cycles
// are not representable in LLVM, we only note the issues when the cycle is
// first formed.

// FIXME: This error is detected early in CodeGen. Once the first error is
// found, Diags.hasErrorOccurred() returs true and we stop the codegen of the
// file. The consequence is that we don't find any subsequent error.

void f1() __attribute__((alias("g1")));
void g1() __attribute__((alias("f1"))); // expected-error {{alias definition is part of a cycle}}

void h1() __attribute__((alias("g1")));
