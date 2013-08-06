// Don't attempt slash switches on msys bash.
// REQUIRES: shell-preserves-root

// Note: %s must be preceded by --, otherwise it may be interpreted as a
// command-line option, e.g. on Mac where %s is commonly under /Users.

// RUN: %clang_cl /TC -### -- %s 2>&1 | FileCheck -check-prefix=TC %s
// TC:  "-x" "c"

// RUN: %clang_cl /TP -### -- %s 2>&1 | FileCheck -check-prefix=TP %s
// TP:  "-x" "c++"

// RUN: %clang_cl -### /Tc%s 2>&1 | FileCheck -check-prefix=Tc %s
// RUN: %clang_cl -### /TP /Tc%s 2>&1 | FileCheck -check-prefix=Tc %s
// Tc:  "-x" "c"

// RUN: %clang_cl -### /Tp%s 2>&1 | FileCheck -check-prefix=Tp %s
// RUN: %clang_cl -### /TC /Tp%s 2>&1 | FileCheck -check-prefix=Tp %s
// Tp:  "-x" "c++"

// RUN: %clang_cl /TP /TC -### -- %s 2>&1 | FileCheck -check-prefix=WARN %s
// WARN: overriding '/TP' option with '/TC'
