// When -fapple-kext is specified, make sure we add -kext to the linker command.

// RUN: %clang -### -fapple-kext %s 2> %t1
// RUN: FileCheck --check-prefix=CHECK1 < %t1 %s

// CHECK1: "-fapple-kext"
// CHECK1: "-kext"

// RUN: %clang -### -fapple-kext -Xlinker -kext %s 2> %t2
// RUN: FileCheck --check-prefix=CHECK2 < %t2 %s

// CHECK2: "-fapple-kext"
// CHECK2: "-kext"
// CHECK2-NOT: "-kext"

