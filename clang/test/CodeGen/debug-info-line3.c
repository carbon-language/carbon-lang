// RUN: %clang_cc1 -g -S -emit-llvm %s -o - | FileCheck %s

// Temporarily XFAIL while investigating regression. (other improvements seem
// more important to keep rather than reverting them in favor of preserving
// this)
// XFAIL: *

void func(char c, char* d)
{
  *d = c + 1;
  return;
  

  
  
  
  
}

// CHECK: ret void, !dbg [[LINE:.*]]
// CHECK: [[LINE]] = !{i32 6,
