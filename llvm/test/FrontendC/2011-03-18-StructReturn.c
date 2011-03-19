// R UN: %llvmgcc %s -S -O0 -o - | FileCheck %s
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


