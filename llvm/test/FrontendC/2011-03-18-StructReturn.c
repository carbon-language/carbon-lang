// RUN: %llvmgcc %s -m64 -S -O0 -o - | FileCheck %s
// XFAIL: *
// XTARGET: darwin
// Radar 9156771
typedef struct RGBColor {
  unsigned short red;
  unsigned short green;
  unsigned short blue;
} RGBColor;

RGBColor func();

RGBColor X;
void foo() {
//CHECK: store i48
  X = func();
}


