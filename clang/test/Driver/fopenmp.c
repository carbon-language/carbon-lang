// RUN: %clang -target x86_64-linux-gnu -fopenmp -c %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-CC1-NO-OPENMP
// RUN: %clang -target x86_64-linux-gnu -fopenmp=libomp -c %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-CC1-OPENMP
// RUN: %clang -target x86_64-linux-gnu -fopenmp=libgomp -c %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-CC1-NO-OPENMP
// RUN: %clang -target x86_64-linux-gnu -fopenmp=libiomp5 -c %s -### 2>&1 | FileCheck %s --check-prefix=CHECK-CC1-OPENMP
//
// CHECK-CC1-OPENMP: "-cc1"
// CHECK-CC1-OPENMP: "-fopenmp"
//
// CHECK-CC1-NO-OPENMP: "-cc1"
// CHECK-CC1-NO-OPENMP-NOT: "-fopenmp"
//
// RUN: %clang -target x86_64-linux-gnu -fopenmp %s -o %t -### 2>&1 | FileCheck %s --check-prefix=CHECK-LD-GOMP
// RUN: %clang -target x86_64-linux-gnu -fopenmp=libomp %s -o %t -### 2>&1 | FileCheck %s --check-prefix=CHECK-LD-OMP
// RUN: %clang -target x86_64-linux-gnu -fopenmp=libgomp %s -o %t -### 2>&1 | FileCheck %s --check-prefix=CHECK-LD-GOMP
// RUN: %clang -target x86_64-linux-gnu -fopenmp=libiomp5 %s -o %t -### 2>&1 | FileCheck %s --check-prefix=CHECK-LD-IOMP5
//
// CHECK-LD-OMP: "{{.*}}ld{{(.exe)?}}"
// CHECK-LD-OMP: "-lomp"
//
// CHECK-LD-GOMP: "{{.*}}ld{{(.exe)?}}"
// CHECK-LD-GOMP: "-lgomp"
//
// CHECK-LD-IOMP5: "{{.*}}ld{{(.exe)?}}"
// CHECK-LD-IOMP5: "-liomp5"
