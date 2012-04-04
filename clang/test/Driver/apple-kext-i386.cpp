// Check that we transparently fallback to llvm-gcc for i386 kexts, we don't
// support the ABI they use (yet).

// RUN: %clang -target i386-apple-darwin10 \
// RUN:   -fapple-kext -### -fsyntax-only %s 2> %t
// RUN: FileCheck --check-prefix=CHECK < %t %s

// CHECK: cc1plus"
// CHECK: "-fapple-kext"

// RUN: %clang -target i386-apple-darwin10 \
// RUN:   -mkernel -### -fsyntax-only %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-MKERNEL < %t %s

// CHECK-MKERNEL: cc1plus"
// CHECK-MKERNEL: "-mkernel"

// RUN: %clang -target i386-apple-darwin10 \
// RUN:   -Wno-self-assign -Wc++11-extensions -Wno-microsoft -Wmicrosoft -Wvla \
// RUN:   -faltivec -mthumb -mcpu=G4 -mlongcall -mno-longcall -msoft-float \
// RUN:   -fapple-kext -### -fsyntax-only %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-UNSUPPORTED < %t %s

// CHECK-UNSUPPORTED: cc1plus"
// CHECK-UNSUPPORTED-NOT: "-Wno-self-assign"
// CHECK-UNSUPPORTED-NOT: "-Wc++11-extensions"
// CHECK-UNSUPPORTED-NOT: "-Wno-microsoft"
// CHECK-UNSUPPORTED-NOT: "-Wmicrosoft"
// CHECK-UNSUPPORTED-NOT: "-Wvla"
// CHECK-UNSUPPORTED-NOT: "-faltivec"
// CHECK-UNSUPPORTED-NOT: "-mthumb"
// CHECK-UNSUPPORTED-NOT: "-mlongcall"
// CHECK-UNSUPPORTED: "-mno-longcall"
// CHECK-UNSUPPORTED: "-msoft-float"

// RUN: %clang -target i386-apple-darwin10 \
// RUN:   -Wconstant-logical-operand -save-temps \
// RUN:   -fapple-kext -### -fsyntax-only %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-UNSUPPORTED2 < %t %s

// CHECK-UNSUPPORTED2: cc1plus"
// CHECK-UNSUPPORTED2-NOT: "-Wconstant-logical-operand"

// Check that --serialize-diagnostics does not cause an "argument unused" error.
// RUN: %clang -target i386-apple-darwin10 \
// RUN:   -Wall -fapple-kext -### --serialize-diagnostics %t.dia -c %s 2>&1 | \
// RUN:   FileCheck --check-prefix=CHECK-UNUSED %s

// CHECK-UNUSED-NOT: argument unused
// CHECK-UNUSED: cc1plus
