// RUN: %clang -ccc-host-triple i386-apple-darwin10 -S -g -dA %s -o - | FileCheck %s
int global;
// CHECK: asciz  "global"                                    ## DW_AT_name
int main() { return 0;}
