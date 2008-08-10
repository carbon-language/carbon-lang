// RUN: clang --emit-llvm -o %t %s &&
// RUN: grep "i8 52" %s | count 1

struct et7 {
  float lv7[0];
  char mv7:6;
} yv7 = {
  {}, 
  52, 
};

