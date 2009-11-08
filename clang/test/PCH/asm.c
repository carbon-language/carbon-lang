// Test this without pch.
// RUN: clang-cc -triple i386-unknown-unknown -include %S/asm.h -fsyntax-only -verify %s

// Test with pch.
// RUN: clang-cc -triple i386-unknown-unknown -emit-pch -o %t %S/asm.h
// RUN: clang-cc -triple i386-unknown-unknown -include-pch %t -fsyntax-only -verify %s 


void call_f(void) { f(); }

void call_clobbers(void) { clobbers(); }
