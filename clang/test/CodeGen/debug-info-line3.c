// RUN: %clang_cc1 -g -S -emit-llvm %s -o - | FileCheck %s

void func(char c, char* d)
{
  *d = c + 1;
  return;
  

  
  
  
  
}

// CHECK: ret void, !dbg !19
// CHECK: !19 = metadata !{i32 6,
