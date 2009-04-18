// Test this without pch.
// RUN: clang-cc -include %S/asm.h -fsyntax-only -verify %s &&

// Test with pch.
// RUN: clang-cc -emit-pch -o %t %S/asm.h &&
// RUN: clang-cc -include-pch %t -fsyntax-only -verify %s 


void call_f(void) { f(); }

void call_clobbers(void) { clobbers(); }
