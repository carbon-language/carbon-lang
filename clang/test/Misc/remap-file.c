// RUN: clang-cc -remap-file="%s;%S/Inputs/remapped-file" -fsyntax-only %s 2>&1 | FileCheck %s

// CHECK: remap-file.c:1:28: warning: incompatible pointer types

int
