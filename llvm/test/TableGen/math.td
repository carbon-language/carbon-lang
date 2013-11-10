// RUN: llvm-tblgen %s | FileCheck %s
// XFAIL: vg_leak

class Int<int value> {
  int Value = value;
}

def v1024   : Int<1024>;
// CHECK: def v1024
// CHECK: Value = 1024

def v1025   : Int<!add(v1024.Value, 1)>;
// CHECK: def v1025
// CHECK: Value = 1025

def v2048   : Int<!add(v1024.Value, v1024.Value)>;
// CHECK: def v2048
// CHECK: Value = 2048

