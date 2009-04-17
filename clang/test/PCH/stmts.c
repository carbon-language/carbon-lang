// Test this without pch.
// RUN: clang-cc -fblocks -include %S/stmts.h -fsyntax-only -emit-llvm -o - %s

// Test with pch.
// RUN: clang-cc -emit-pch -fblocks -o %t %S/stmts.h &&
// RUN: clang-cc -fblocks -include-pch %t -fsyntax-only -emit-llvm -o - %s 

void g0(void) { f0(5); }
