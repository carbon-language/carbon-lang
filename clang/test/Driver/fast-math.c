// Test that the GCC fast-math floating point flags get lowered to the correct
// permutation of Clang frontend flags. This is non-trivial for a few reasons.
// First, the GCC flags have many different and surprising effects. Second,
// LLVM only supports three switches which is more coarse grained than GCC's
// support.
//
// RUN: %clang -### -fno-honor-infinities -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-NO-INFS %s
// CHECK-NO-INFS: "-cc1"
// CHECK-NO-INFS: "-menable-no-infs"
//
// RUN: %clang -### -fno-honor-nans -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-NO-NANS %s
// CHECK-NO-NANS: "-cc1"
// CHECK-NO-NANS: "-menable-no-nans"
//
// RUN: %clang -### -fmath-errno -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-MATH-ERRNO %s
// CHECK-MATH-ERRNO: "-cc1"
// CHECK-MATH-ERRNO: "-fmath-errno"
//
// RUN: %clang -### -fmath-errno -fno-math-errno -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-NO-MATH-ERRNO %s
// RUN: %clang -### -target i686-apple-darwin -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-NO-MATH-ERRNO %s
// CHECK-NO-MATH-ERRNO: "-cc1"
// CHECK-NO-MATH-ERRNO-NOT: "-fmath-errno"
//
// RUN: %clang -### -fno-math-errno -fassociative-math -freciprocal-math \
// RUN:     -fno-signed-zeros -fno-trapping-math -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-UNSAFE-MATH %s
// CHECK-UNSAFE-MATH: "-cc1"
// CHECK-UNSAFE-MATH: "-menable-unsafe-fp-math"
//
// Check that various umbrella flags also enable these frontend options.
// RUN: %clang -### -ffast-math -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-NO-INFS %s
// RUN: %clang -### -ffast-math -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-NO-NANS %s
// RUN: %clang -### -ffast-math -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-UNSAFE-MATH %s
// RUN: %clang -### -ffinite-math-only -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-NO-INFS %s
// RUN: %clang -### -ffinite-math-only -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-NO-NANS %s
// RUN: %clang -### -funsafe-math-optimizations -fno-math-errno -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-UNSAFE-MATH %s
//
// One umbrella flag is *really* weird and also changes the semantics of the
// program by adding a special preprocessor macro. Check that the frontend flag
// modeling this semantic change is provided. Also check that the semantic
// impact remains even if every optimization is disabled.
// RUN: %clang -### -ffast-math -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-FAST-MATH %s
// RUN: %clang -### -ffast-math -fno-finite-math-only \
// RUN:     -fno-unsafe-math-optimizations -fmath-errno -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-FAST-MATH %s
// CHECK-FAST-MATH: "-cc1"
// CHECK-FAST-MATH: "-ffast-math"
//
// Check various means of disabling these flags, including disabling them after
// they've been enabled via an umbrella flag.
// RUN: %clang -### -fno-honor-infinities -fhonor-infinities -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-NO-NO-INFS %s
// RUN: %clang -### -ffinite-math-only -fhonor-infinities -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-NO-NO-INFS %s
// RUN: %clang -### -ffinite-math-only -fno-finite-math-only -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-NO-NO-INFS %s
// RUN: %clang -### -ffast-math -fhonor-infinities -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-NO-NO-INFS %s
// RUN: %clang -### -ffast-math -fno-finite-math-only -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-NO-NO-INFS %s
// CHECK-NO-NO-INFS: "-cc1"
// CHECK-NO-NO-INFS-NOT: "-menable-no-infs"
// CHECK-NO-NO-INFS: "-o"
//
// RUN: %clang -### -fno-honor-nans -fhonor-nans -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-NO-NO-NANS %s
// RUN: %clang -### -ffinite-math-only -fhonor-nans -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-NO-NO-NANS %s
// RUN: %clang -### -ffinite-math-only -fno-finite-math-only -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-NO-NO-NANS %s
// RUN: %clang -### -ffast-math -fhonor-nans -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-NO-NO-NANS %s
// RUN: %clang -### -ffast-math -fno-finite-math-only -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-NO-NO-NANS %s
// CHECK-NO-NO-NANS: "-cc1"
// CHECK-NO-NO-NANS-NOT: "-menable-no-nans"
// CHECK-NO-NO-NANS: "-o"
//
// RUN: %clang -### -fassociative-math -freciprocal-math -fno-signed-zeros \
// RUN:     -fno-trapping-math -fno-associative-math -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-NO-UNSAFE-MATH %s
// RUN: %clang -### -fassociative-math -freciprocal-math -fno-signed-zeros \
// RUN:     -fno-trapping-math -fno-reciprocal-math -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-NO-UNSAFE-MATH %s
// RUN: %clang -### -fassociative-math -freciprocal-math -fno-signed-zeros \
// RUN:     -fno-trapping-math -fsigned-zeros -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-NO-UNSAFE-MATH %s
// RUN: %clang -### -fassociative-math -freciprocal-math -fno-signed-zeros \
// RUN:     -fno-trapping-math -ftrapping-math -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-NO-UNSAFE-MATH %s
// RUN: %clang -### -funsafe-math-optimizations -fno-associative-math -c %s \
// RUN:   2>&1 | FileCheck --check-prefix=CHECK-NO-UNSAFE-MATH %s
// RUN: %clang -### -funsafe-math-optimizations -fno-reciprocal-math -c %s \
// RUN:   2>&1 | FileCheck --check-prefix=CHECK-NO-UNSAFE-MATH %s
// RUN: %clang -### -funsafe-math-optimizations -fsigned-zeros -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-NO-UNSAFE-MATH %s
// RUN: %clang -### -funsafe-math-optimizations -ftrapping-math -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-NO-UNSAFE-MATH %s
// RUN: %clang -### -funsafe-math-optimizations -fno-unsafe-math-optimizations \
// RUN:     -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-NO-UNSAFE-MATH %s
// RUN: %clang -### -ffast-math -fno-associative-math -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-NO-UNSAFE-MATH %s
// RUN: %clang -### -ffast-math -fno-reciprocal-math -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-NO-UNSAFE-MATH %s
// RUN: %clang -### -ffast-math -fsigned-zeros -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-NO-UNSAFE-MATH %s
// RUN: %clang -### -ffast-math -ftrapping-math -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-NO-UNSAFE-MATH %s
// RUN: %clang -### -ffast-math -fno-unsafe-math-optimizations -c %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-NO-UNSAFE-MATH %s
// CHECK-NO-UNSAFE-MATH: "-cc1"
// CHECK-NO-UNSAFE-MATH-NOT: "-menable-unsafe-fp-math"
// CHECK-NO-UNSAFE-MATH: "-o"
