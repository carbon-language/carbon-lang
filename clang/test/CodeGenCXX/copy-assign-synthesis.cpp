// RUN: clang-cc -emit-llvm -o %t %s &&
// RUN: grep "_ZN1XaSERK1X" %t | count 0

extern "C" int printf(...);

struct X { 
  X() : d(0.0), d1(1.1), d2(1.2), d3(1.3) {}
  double d;
  double d1;
  double d2;
  double d3;
  void pr() {
    printf("d = %f d1 = %f d2 = %f d3 = %f\n", d, d1,d2,d3);
  }
}; 


X srcX; 
X dstX; 
X dstY; 

int main() {
  dstY = dstX = srcX;
  srcX.pr();
  dstX.pr();
  dstY.pr();
}

