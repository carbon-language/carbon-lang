// Don't attempt slash switches on msys bash.
// REQUIRES: shell-preserves-root
// REQUIRES: x86-registered-target

// We support -m32 and -m64.  We support all x86 CPU feature flags in gcc's -m
// flag space.
// RUN: %clang_cl /Zs /WX -m32 -m64 -msse3 -msse4.1 -mavx -mno-avx \
// RUN:     -### -- 2>&1 %s | FileCheck -check-prefix=MFLAGS %s
// MFLAGS-NOT: argument unused during compilation

void f() {
}
