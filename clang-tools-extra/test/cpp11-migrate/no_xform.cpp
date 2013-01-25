// RUN: grep -Ev "// *[A-Z-]+:" %s > %t.cpp
// RUN: cpp11-migrate %t.cpp --
// XFAIL: *

int main(int argc, char** argv) {
  return 0;
}
