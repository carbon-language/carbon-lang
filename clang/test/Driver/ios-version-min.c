// RUN: %clang -target i386-apple-darwin9 -miphonesimulator-version-min=7.0 -arch i386 -S -o - %s | FileCheck %s
// RUN: %clang -target i386-apple-darwin9 -miphoneos-version-min=7.0 -arch armv7s -S -o - %s | FileCheck %s

int main() { return 0; }
// CHECK: .ios_version_min 7, 0
