// Test the that the driver produces reasonable linker invocations with
// -fopenmp.
//
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     -fopenmp -target i386-unknown-linux \
// RUN:   | FileCheck --check-prefix=CHECK-LD-32 %s
// CHECK-LD-32: "{{.*}}ld{{(.exe)?}}"
// CHECK-LD-32: "-lgomp" "-lrt" "-lgcc"
// CHECK-LD-32: "-lpthread" "-lc"
//
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     -fopenmp -target x86_64-unknown-linux \
// RUN:   | FileCheck --check-prefix=CHECK-LD-64 %s
// CHECK-LD-64: "{{.*}}ld{{(.exe)?}}"
// CHECK-LD-64: "-lgomp" "-lrt" "-lgcc"
// CHECK-LD-64: "-lpthread" "-lc"
