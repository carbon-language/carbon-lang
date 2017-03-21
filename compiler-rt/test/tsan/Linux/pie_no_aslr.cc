// RUN: %clang_tsan %s -pie -fPIE -o %t && %run setarch x86_64 -R %t

int main() {
  return 0;
}
