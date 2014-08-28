// Check that the name of the arch we bind is "ppc" not "powerpc".
//
// RUN: %clang -target powerpc-apple-darwin8 -### \
// RUN:   -ccc-print-phases %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-BIND-PPC < %t %s
//
// CHECK-BIND-PPC: bind-arch, "ppc"
//
// RUN: %clang -target powerpc64-apple-darwin8 -### \
// RUN:   -ccc-print-phases %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-BIND-PPC64 < %t %s
//
// CHECK-BIND-PPC64: bind-arch, "ppc64"

// Check that the correct arch name is passed to the external assembler
//
// RUN: %clang -target powerpc-apple-darwin8 -### \
// RUN:   -no-integrated-as -c %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-AS-PPC < %t %s
//
// CHECK-AS-PPC: {{as(.exe)?"}}
// CHECK-AS-PPC: "-arch" "ppc"
//
// RUN: %clang -target powerpc64-apple-darwin8 -### \
// RUN:   -no-integrated-as -c %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-AS-PPC64 < %t %s
//
// CHECK-AS-PPC64: {{as(.exe)?"}}
// CHECK-AS-PPC64: "-arch" "ppc64"

// Check that the correct arch name is passed to the external linker
//
// RUN: %clang -target powerpc-apple-darwin8 -### %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-LD-PPC < %t %s
//
// CHECK-LD-PPC: {{ld(.exe)?"}}
// CHECK-LD-PPC: "-arch" "ppc"
//
// RUN: %clang -target powerpc64-apple-darwin8 -### %s 2> %t
// RUN: FileCheck --check-prefix=CHECK-LD-PPC64 < %t %s
//
// CHECK-LD-PPC64: {{ld(.exe)?"}}
// CHECK-LD-PPC64: "-arch" "ppc64"
