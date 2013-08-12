// Don't attempt slash switches on msys bash.
// REQUIRES: shell-preserves-root

// Note: %s must be preceded by --, otherwise it may be interpreted as a
// command-line option, e.g. on Mac where %s is commonly under /Users.


// RUN: %clang_cl -### -- %s 2>&1 | FileCheck -check-prefix=DEFAULTEXE %s
// DEFAULTEXE: cl-Fe.exe

// RUN: %clang_cl /Fefoo -### -- %s 2>&1 | FileCheck -check-prefix=FeNOEXT %s
// FeNOEXT: foo.exe

// RUN: %clang_cl /Fefoo.ext -### -- %s 2>&1 | FileCheck -check-prefix=FeEXT %s
// FeEXT: foo.ext

// RUN: %clang_cl /Fefoo.dir/ -### -- %s 2>&1 | FileCheck -check-prefix=FeDIR %s
// FeDIR: foo.dir{{[/\\]+}}cl-Fe.exe

// RUN: %clang_cl /Fefoo.dir/a -### -- %s 2>&1 | FileCheck -check-prefix=FeDIRNAME %s
// FeDIRNAME: foo.dir{{[/\\]+}}a.exe

// RUN: %clang_cl /Fefoo.dir/a.ext -### -- %s 2>&1 | FileCheck -check-prefix=FeDIRNAMEEXT %s
// FeDIRNAMEEXT: foo.dir{{[/\\]+}}a.ext

// RUN: %clang_cl /Fe -### 2>&1 | FileCheck -check-prefix=MISSINGARG %s
// MISSINGARG: error: argument to '/Fe' is missing (expected 1 value)

// RUN: %clang_cl /Fefoo /Febar -### -- %s 2>&1 | FileCheck -check-prefix=FEOVERRIDE %s
// FEOVERRIDE: warning: overriding '/Fefoo' option with '/Febar'
// FEOVERRIDE: bar.exe
