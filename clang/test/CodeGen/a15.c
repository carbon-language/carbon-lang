// RUN: %clang -ccc-host-triple armv7-none-linux-gnueabi -mcpu=cortex-a15 -emit-llvm -S %s  -o /dev/null
// REQUIRES: arm-registered-target

int main() {
  return 0;
}
