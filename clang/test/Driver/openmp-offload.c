///
/// Perform several driver tests for OpenMP offloading
///

/// ###########################################################################

/// Check whether an invalid OpenMP target is specified:
// RUN:   %clang -### -fopenmp=libomp -fopenmp-targets=aaa-bbb-ccc-ddd %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-INVALID-TARGET %s
// RUN:   %clang -### -fopenmp -fopenmp-targets=aaa-bbb-ccc-ddd %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-INVALID-TARGET %s
// CHK-INVALID-TARGET: error: OpenMP target is invalid: 'aaa-bbb-ccc-ddd'

/// ###########################################################################

/// Check warning for empty -fopenmp-targets
// RUN:   %clang -### -fopenmp=libomp -fopenmp-targets=  %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-EMPTY-OMPTARGETS %s
// RUN:   %clang -### -fopenmp -fopenmp-targets=  %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-EMPTY-OMPTARGETS %s
// CHK-EMPTY-OMPTARGETS: warning: joined argument expects additional value: '-fopenmp-targets='

/// ###########################################################################

/// Check error for no -fopenmp option
// RUN:   %clang -### -fopenmp-targets=powerpc64le-ibm-linux-gnu  %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-NO-FOPENMP %s
// RUN:   %clang -### -fopenmp=libgomp -fopenmp-targets=powerpc64le-ibm-linux-gnu  %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-NO-FOPENMP %s
// CHK-NO-FOPENMP: error: The option -fopenmp-targets must be used in conjunction with a -fopenmp option compatible with offloading, please use -fopenmp=libomp or -fopenmp=libiomp5.

/// ###########################################################################

/// Check warning for duplicate offloading targets.
// RUN:   %clang -### -ccc-print-phases -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu,powerpc64le-ibm-linux-gnu  %s 2>&1 \
// RUN:   | FileCheck -check-prefix=CHK-DUPLICATES %s
// CHK-DUPLICATES: warning: The OpenMP offloading target 'powerpc64le-ibm-linux-gnu' is similar to target 'powerpc64le-ibm-linux-gnu' already specified - will be ignored.
