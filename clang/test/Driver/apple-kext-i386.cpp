// Check that we transparently fallback to llvm-gcc for i386 kexts, we don't
// support the ABI they use (yet).

// RUN: %clang -ccc-host-triple i386-apple-darwin10 \
// RUN:   -fapple-kext -### -fsyntax-only %s 2> %t
// RUN: FileCheck --check-prefix=CHECK < %t %s

// CHECK: cc1plus"
// CHECK: "-fapple-kext"

// RUN: %clang -ccc-host-triple i386-apple-darwin10 \
// RUN:   -mkernel -### -fsyntax-only %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-MKERNEL < %t %s

// CHECK-MKERNEL: cc1plus"
// CHECK-MKERNEL: "-mkernel"

// RUN: %clang -ccc-host-triple i386-apple-darwin10 \
// RUN:   -Wno-self-assign -Wc++0x-extensions -Wno-microsoft -Wmicrosoft -Wvla \
// RUN:   -faltivec -mthumb -mcpu=G4 -mlongcall -mno-longcall -msoft-float \
// RUN:   -fapple-kext -### -fsyntax-only %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-UNSUPPORTED < %t %s

// CHECK-UNSUPPORTED: cc1plus"
// CHECK-UNSUPPORTED-NOT: "-Wno-self-assign"
// CHECK-UNSUPPORTED-NOT: "-Wc++0x-extensions"
// CHECK-UNSUPPORTED-NOT: "-Wno-microsoft"
// CHECK-UNSUPPORTED-NOT: "-Wmicrosoft"
// CHECK-UNSUPPORTED-NOT: "-Wvla"
// CHECK-UNSUPPORTED-NOT: "-faltivec"
// CHECK-UNSUPPORTED-NOT: "-mthumb"
// CHECK-UNSUPPORTED-NOT: "-mlongcall"
// CHECK-UNSUPPORTED: "-mno-longcall"
// CHECK-UNSUPPORTED: "-msoft-float"

