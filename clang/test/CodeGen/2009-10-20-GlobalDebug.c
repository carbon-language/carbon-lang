// RUN: clang -S -g -dA %s -o - | FileCheck %s
int global;
// CHECK: asciz  "global"                                    ## DW_AT_MIPS_linkage_name
int main() { return 0;}
