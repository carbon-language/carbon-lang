// RUN: grep -Ev "// *[A-Z-]+:" %s > %t.cpp
// RUN: cpp11-migrate %t.cpp --
// XFAIL: *
// REQUIRES: shell

int main(int argc, char** argv) {
i  return 0;
}
