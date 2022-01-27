// RUN: %clang -S -### -cl-std=CL %s 2>&1 | FileCheck --check-prefix=CHECK-CL %s
// RUN: %clang -S -### -cl-std=CL1.0 %s 2>&1 | FileCheck --check-prefix=CHECK-CL10 %s
// RUN: %clang -S -### -cl-std=CL1.1 %s 2>&1 | FileCheck --check-prefix=CHECK-CL11 %s
// RUN: %clang -S -### -cl-std=CL1.2 %s 2>&1 | FileCheck --check-prefix=CHECK-CL12 %s
// RUN: %clang -S -### -cl-std=CL2.0 %s 2>&1 | FileCheck --check-prefix=CHECK-CL20 %s
// RUN: %clang -S -### -cl-std=CL3.0 %s 2>&1 | FileCheck --check-prefix=CHECK-CL30 %s
// RUN: %clang -S -### -cl-std=clc++ %s 2>&1 | FileCheck --check-prefix=CHECK-CLCPP %s
// RUN: %clang -S -### -cl-opt-disable %s 2>&1 | FileCheck --check-prefix=CHECK-OPT-DISABLE %s
// RUN: %clang -S -### -cl-strict-aliasing %s 2>&1 | FileCheck --check-prefix=CHECK-STRICT-ALIASING %s
// RUN: %clang -S -### -cl-single-precision-constant %s 2>&1 | FileCheck --check-prefix=CHECK-SINGLE-PRECISION-CONST %s
// RUN: %clang -S -### -cl-finite-math-only %s 2>&1 | FileCheck --check-prefix=CHECK-FINITE-MATH-ONLY %s
// RUN: %clang -S -### -cl-kernel-arg-info %s 2>&1 | FileCheck --check-prefix=CHECK-KERNEL-ARG-INFO %s
// RUN: %clang -S -### -cl-unsafe-math-optimizations %s 2>&1 | FileCheck --check-prefix=CHECK-UNSAFE-MATH-OPT %s
// RUN: %clang -S -### -cl-fast-relaxed-math %s 2>&1 | FileCheck --check-prefix=CHECK-FAST-RELAXED-MATH %s
// RUN: %clang -S -### -cl-mad-enable %s 2>&1 | FileCheck --check-prefix=CHECK-MAD-ENABLE %s
// RUN: %clang -S -### -cl-no-signed-zeros %s 2>&1 | FileCheck --check-prefix=CHECK-NO-SIGNED-ZEROS %s
// RUN: %clang -S -### -cl-denorms-are-zero %s 2>&1 | FileCheck --check-prefix=CHECK-DENORMS-ARE-ZERO %s
// RUN: %clang -S -### -cl-fp32-correctly-rounded-divide-sqrt %s 2>&1 | FileCheck --check-prefix=CHECK-ROUND-DIV %s
// RUN: %clang -S -### -cl-uniform-work-group-size %s 2>&1 | FileCheck --check-prefix=CHECK-UNIFORM-WG %s
// RUN: not %clang -cl-std=c99 -DOPENCL %s 2>&1 | FileCheck --check-prefix=CHECK-C99 %s
// RUN: not %clang -cl-std=invalid -DOPENCL %s 2>&1 | FileCheck --check-prefix=CHECK-INVALID %s
// RUN: %clang -S -### -target spir-unknown-unknown %s 2>&1 | FileCheck --check-prefix=CHECK-W-SPIR-COMPAT %s
// RUN: %clang -S -### -target amdgcn-amd-amdhsa-opencl %s 2>&1 | FileCheck --check-prefix=CHECK-NO-W-SPIR-COMPAT %s

// CHECK-CL: "-cc1" {{.*}} "-cl-std=CL"
// CHECK-CL10: "-cc1" {{.*}} "-cl-std=CL1.0"
// CHECK-CL11: "-cc1" {{.*}} "-cl-std=CL1.1"
// CHECK-CL12: "-cc1" {{.*}} "-cl-std=CL1.2"
// CHECK-CL20: "-cc1" {{.*}} "-cl-std=CL2.0"
// CHECK-CL30: "-cc1" {{.*}} "-cl-std=CL3.0"
// CHECK-CLCPP: "-cc1" {{.*}} "-cl-std=clc++"
// CHECK-OPT-DISABLE: "-cc1" {{.*}} "-cl-opt-disable"
// CHECK-STRICT-ALIASING: "-cc1" {{.*}} "-cl-strict-aliasing"
// CHECK-SINGLE-PRECISION-CONST: "-cc1" {{.*}} "-cl-single-precision-constant"
// CHECK-FINITE-MATH-ONLY: "-cc1" {{.*}} "-cl-finite-math-only"
// CHECK-KERNEL-ARG-INFO: "-cc1" {{.*}} "-cl-kernel-arg-info"
// CHECK-UNSAFE-MATH-OPT: "-cc1" {{.*}} "-cl-unsafe-math-optimizations"
// CHECK-FAST-RELAXED-MATH: "-cc1" {{.*}} "-cl-fast-relaxed-math"
// CHECK-MAD-ENABLE: "-cc1" {{.*}} "-cl-mad-enable"
// CHECK-NO-SIGNED-ZEROS: "-cc1" {{.*}} "-cl-no-signed-zeros"

// This is not forwarded
// CHECK-DENORMS-ARE-ZERO-NOT: "-cl-denorms-are-zero"

// CHECK-ROUND-DIV: "-cc1" {{.*}} "-cl-fp32-correctly-rounded-divide-sqrt"
// CHECK-UNIFORM-WG: "-cc1" {{.*}} "-cl-uniform-work-group-size"
// CHECK-C99: error: invalid value 'c99' in '-cl-std=c99'
// CHECK-INVALID: error: invalid value 'invalid' in '-cl-std=invalid'

// CHECK-W-SPIR-COMPAT: "-Wspir-compat"
// CHECK-NO-W-SPIR-COMPAT-NOT: "-Wspir-compat"

kernel void func(void);
