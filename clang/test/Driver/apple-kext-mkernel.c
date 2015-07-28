// RUN: %clang -target x86_64-apple-darwin10 \
// RUN:   -mkernel -### -fsyntax-only %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-X86 < %t %s

// CHECK-X86: "-disable-red-zone"
// CHECK-X86: "-fno-builtin"
// CHECK-X86: "-fno-rtti"
// CHECK-X86: "-fno-common"

// RUN: %clang -target x86_64-apple-darwin10 \
// RUN:   -arch armv7 -mkernel -mstrict-align -### -fsyntax-only %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-ARM < %t %s

// CHECK-ARM: "-target-feature" "+long-calls"
// CHECK-ARM: "-target-feature" "+strict-align"
// CHECK-ARM-NOT: "-target-feature" "+strict-align"
// CHECK-ARM: "-fno-builtin"
// CHECK-ARM: "-fno-rtti"
// CHECK-ARM: "-fno-common"

// RUN: %clang -target x86_64-apple-darwin10 \
// RUN:   -Werror -fno-builtin -fno-exceptions -fno-common -fno-rtti \
// RUN:   -mkernel -fsyntax-only %s
