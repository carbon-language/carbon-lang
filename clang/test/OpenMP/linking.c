// Test the that the driver produces reasonable linker invocations with
// -fopenmp or -fopenmp=libiomp5|libgomp.
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
//
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     -fopenmp=libgomp -target i386-unknown-linux \
// RUN:   | FileCheck --check-prefix=CHECK-GOMP-LD-32 %s
// CHECK-GOMP-LD-32: "{{.*}}ld{{(.exe)?}}"
// CHECK-GOMP-LD-32: "-lgomp" "-lrt" "-lgcc"
// CHECK-GOMP-LD-32: "-lpthread" "-lc"
//
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     -fopenmp=libgomp -target x86_64-unknown-linux \
// RUN:   | FileCheck --check-prefix=CHECK-GOMP-LD-64 %s
// CHECK-GOMP-LD-64: "{{.*}}ld{{(.exe)?}}"
// CHECK-GOMP-LD-64: "-lgomp" "-lrt" "-lgcc"
// CHECK-GOMP-LD-64: "-lpthread" "-lc"
//
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     -fopenmp=libiomp5 -target i386-unknown-linux \
// RUN:   | FileCheck --check-prefix=CHECK-IOMP5-LD-32 %s
// CHECK-IOMP5-LD-32: "{{.*}}ld{{(.exe)?}}"
// CHECK-IOMP5-LD-32: "-liomp5" "-lgcc"
// CHECK-IOMP5-LD-32: "-lpthread" "-lc"
//
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     -fopenmp=libiomp5 -target x86_64-unknown-linux \
// RUN:   | FileCheck --check-prefix=CHECK-IOMP5-LD-64 %s
// CHECK-IOMP5-LD-64: "{{.*}}ld{{(.exe)?}}"
// CHECK-IOMP5-LD-64: "-liomp5" "-lgcc"
// CHECK-IOMP5-LD-64: "-lpthread" "-lc"
//
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     -fopenmp=lib -target i386-unknown-linux \
// RUN:   | FileCheck --check-prefix=CHECK-LIB-LD-32 %s
// CHECK-LIB-LD-32: error: unsupported argument 'lib' to option 'fopenmp='
//
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     -fopenmp=lib -target x86_64-unknown-linux \
// RUN:   | FileCheck --check-prefix=CHECK-LIB-LD-64 %s
// CHECK-LIB-LD-64: error: unsupported argument 'lib' to option 'fopenmp='
//
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     -fopenmp -fopenmp=libiomp5 -target i386-unknown-linux \
// RUN:   | FileCheck --check-prefix=CHECK-LD-WARN-32 %s
// CHECK-LD-WARN-32: warning: argument unused during compilation: '-fopenmp=libiomp5'
// CHECK-LD-WARN-32: "{{.*}}ld{{(.exe)?}}"
// CHECK-LD-WARN-32: "-lgomp" "-lrt" "-lgcc"
// CHECK-LD-WARN-32: "-lpthread" "-lc"
//
// RUN: %clang -no-canonical-prefixes %s -### -o %t.o 2>&1 \
// RUN:     -fopenmp -fopenmp=libiomp5 -target x86_64-unknown-linux \
// RUN:   | FileCheck --check-prefix=CHECK-LD-WARN-64 %s
// CHECK-LD-WARN-64: warning: argument unused during compilation: '-fopenmp=libiomp5'
// CHECK-LD-WARN-64: "{{.*}}ld{{(.exe)?}}"
// CHECK-LD-WARN-64: "-lgomp" "-lrt" "-lgcc"
// CHECK-LD-WARN-64: "-lpthread" "-lc"
//
