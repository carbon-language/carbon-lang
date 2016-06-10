// Android's target triples can contain a version number in the environment
// field (e.g. arm-linux-androideabi9).
// Make sure that any version is stripped when finding toolchain binaries.

// RUN: env "PATH=%S/Inputs/android_triple_version/bin" \
// RUN:     %clang -### -target arm-linux-androideabi %s 2>&1 | FileCheck %s
// RUN: env "PATH=%S/Inputs/android_triple_version/bin" \
// RUN:     %clang -### -target arm-linux-androideabi9 %s 2>&1 | FileCheck %s

// CHECK: arm-linux-androideabi-ld
