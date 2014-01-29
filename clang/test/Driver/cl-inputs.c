// Don't attempt slash switches on msys bash.
// REQUIRES: shell-preserves-root

// Note: %s must be preceded by --, otherwise it may be interpreted as a
// command-line option, e.g. on Mac where %s is commonly under /Users.

// RUN: %clang_cl /TC -### -- %s 2>&1 | FileCheck -check-prefix=TC %s
// TC:  "-x" "c"
// TC-NOT: warning
// TC-NOT: note

// RUN: %clang_cl /TP -### -- %s 2>&1 | FileCheck -check-prefix=TP %s
// TP:  "-x" "c++"
// TP-NOT: warning
// TP-NOT: note

// RUN: %clang_cl -### /Tc%s /TP -- %s 2>&1 | FileCheck -check-prefix=Tc %s
// RUN: %clang_cl -### /TP /Tc%s -- %s 2>&1 | FileCheck -check-prefix=Tc %s
// Tc:  "-x" "c"
// Tc:  "-x" "c++"
// Tc-NOT: warning
// Tc-NOT: note

// RUN: %clang_cl -### /Tp%s /TC -- %s 2>&1 | FileCheck -check-prefix=Tp %s
// RUN: %clang_cl -### /TC /Tp%s -- %s 2>&1 | FileCheck -check-prefix=Tp %s
// Tp:  "-x" "c++"
// Tp:  "-x" "c"
// Tp-NOT: warning
// Tp-NOT: note

// RUN: %clang_cl /TP /TC /TP -### -- %s 2>&1 | FileCheck -check-prefix=WARN %s
// WARN: warning: overriding '/TP' option with '/TC'
// WARN: warning: overriding '/TC' option with '/TP'
// WARN: note: The last /TC or /TP option takes precedence over earlier instances
// WARN-NOT: note

// RUN: not %clang_cl - 2>&1 | FileCheck -check-prefix=STDIN %s
// STDIN: error: use /Tc or /Tp

// RUN: %clang_cl -### /Tc - 2>&1 | FileCheck -check-prefix=STDINTc %s
// STDINTc: "-x" "c"

void f();
