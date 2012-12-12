// RUN: grep -Ev "// *[A-Z-]+:" %s > %t.cpp
// RUN: cpp11-migrate %t.cpp --
// REQUIRES: shell

int main(int argc, char** argv) {
  return 0;
}
