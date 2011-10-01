// General tests that ld invocations on Linux targets sane.
//
// RUN: %clang -no-canonical-prefixes -ccc-host-triple i386-unknown-linux %s -### -o %t.o 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-LD-32 %s
// CHECK-LD-32: "{{.*}}ld{{(.exe)?}}" {{.*}} "-L/lib/../lib{{(32)?}}" "-L/usr/lib/../lib{{(32)?}}"
//
// RUN: %clang -no-canonical-prefixes -ccc-host-triple x86_64-unknown-linux %s -### -o %t.o 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-LD-64 %s
// CHECK-LD-64: "{{.*}}ld{{(.exe)?}}" {{.*}} "-L/lib/../lib{{(64)?}}" "-L/usr/lib/../lib{{(64)?}}"
