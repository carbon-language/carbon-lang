// General tests that ld invocations on Linux targets sane.
//
// RUN: %clang -no-canonical-prefixes -ccc-host-triple i386-unknown-linux %s -### -o %t.o 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-LD-32 %s
// 32-bit machines may use 'lib' or 'lib32' dependending on how they setup multilib.
// CHECK-LD-32: "{{.*}}ld" {{.*}} "-L/lib/../lib{{32|}}" "-L/usr/lib/../lib{{32|}}"
//
// RUN: %clang -no-canonical-prefixes -ccc-host-triple x86_64-unknown-linux %s -### -o %t.o 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-LD-64 %s
// FIXME: Should we allow 'lib' instead of 'lib64' here?
// CHECK-LD-64: "{{.*}}ld" {{.*}} "-L/lib/../lib64" "-L/usr/lib/../lib64"
