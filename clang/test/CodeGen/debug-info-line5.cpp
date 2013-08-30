// RUN: %clang %s -g -gcolumn-info -S -emit-llvm -o - | FileCheck %s
// Line table entries should reference this cpp file, not the header

#include "debug-info-line5.h"

int result;
int foo(unsigned);

int main()
{
  while ( 1 )
  {
    result = foo(Marker);
  }
  return result;
}

// CHECK: !{{[0-9]*}} = metadata !{i32 {{[0-9]*}}, i32 {{[0-9]*}}, null, metadata !"Marker", {{.*}} ; [ DW_TAG_variable ] [Marker]
