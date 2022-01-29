// Make sure unbounded arrays compile with debug information.
//
// RUN: %clang_cc1 -emit-llvm -debug-info-kind=limited %s -o -

// PR1068

struct Object {
  char buffer[];
};

int main(int argc, char** argv) {
  new Object;
  return 0;
}
