// Don't attempt slash switches on msys bash.
// REQUIRES: shell-preserves-root

// Note: %s must be preceded by --, otherwise it may be interpreted as a
// command-line option, e.g. on Mac where %s is commonly under /Users.

// RUN: %clang_cl /Foa -### -- %s 2>&1 | FileCheck -check-prefix=FoNAME %s
// FoNAME:  "-o" "a.obj"

// RUN: %clang_cl /Foa.ext /Fob.ext -### -- %s 2>&1 | FileCheck -check-prefix=FoNAMEEXT %s
// FoNAMEEXT:  warning: overriding '/Foa.ext' option with '/Fob.ext'
// FoNAMEEXT:  "-o" "b.ext"

// RUN: %clang_cl /Fofoo.dir/ -### -- %s 2>&1 | FileCheck -check-prefix=FoDIR %s
// FoDIR:  "-o" "foo.dir{{[/\\]+}}cl-outputs.obj"

// RUN: %clang_cl /Fofoo.dir/a -### -- %s 2>&1 | FileCheck -check-prefix=FoDIRNAME %s
// FoDIRNAME:  "-o" "foo.dir{{[/\\]+}}a.obj"

// RUN: %clang_cl /Fofoo.dir/a.ext -### -- %s 2>&1 | FileCheck -check-prefix=FoDIRNAMEEXT %s
// FoDIRNAMEEXT:  "-o" "foo.dir{{[/\\]+}}a.ext"

// RUN: %clang_cl /Fo.. -### -- %s 2>&1 | FileCheck -check-prefix=FoCRAZY %s
// FoCRAZY:  "-o" "..obj"

// RUN: %clang_cl /Fo -### 2>&1 | FileCheck -check-prefix=FoMISSINGARG %s
// FoMISSINGARG: error: argument to '/Fo' is missing (expected 1 value)

// RUN: %clang_cl /Foa.obj -### -- %s %s 2>&1 | FileCheck -check-prefix=CHECK-MULTIPLESOURCEERROR %s
// CHECK-MULTIPLESOURCEERROR: error: cannot specify '/Foa.obj' when compiling multiple source files

// RUN: %clang_cl /Fomydir/ -### -- %s %s 2>&1 | FileCheck -check-prefix=CHECK-MULTIPLESOURCEOK %s
// CHECK-MULTIPLESOURCEOK: "-o" "mydir{{[/\\]+}}cl-outputs.obj"


// RUN: %clang_cl -### -- %s 2>&1 | FileCheck -check-prefix=DEFAULTEXE %s
// DEFAULTEXE: cl-outputs.exe

// RUN: %clang_cl /Fefoo -### -- %s 2>&1 | FileCheck -check-prefix=FeNOEXT %s
// FeNOEXT: foo.exe

// RUN: %clang_cl /Fefoo.ext -### -- %s 2>&1 | FileCheck -check-prefix=FeEXT %s
// FeEXT: foo.ext

// RUN: %clang_cl /Fefoo.dir/ -### -- %s 2>&1 | FileCheck -check-prefix=FeDIR %s
// FeDIR: foo.dir{{[/\\]+}}cl-outputs.exe

// RUN: %clang_cl /Fefoo.dir/a -### -- %s 2>&1 | FileCheck -check-prefix=FeDIRNAME %s
// FeDIRNAME: foo.dir{{[/\\]+}}a.exe

// RUN: %clang_cl /Fefoo.dir/a.ext -### -- %s 2>&1 | FileCheck -check-prefix=FeDIRNAMEEXT %s
// FeDIRNAMEEXT: foo.dir{{[/\\]+}}a.ext

// RUN: %clang_cl /Fe -### 2>&1 | FileCheck -check-prefix=FeMISSINGARG %s
// FeMISSINGARG: error: argument to '/Fe' is missing (expected 1 value)

// RUN: %clang_cl /Fefoo /Febar -### -- %s 2>&1 | FileCheck -check-prefix=FeOVERRIDE %s
// FeOVERRIDE: warning: overriding '/Fefoo' option with '/Febar'
// FeOVERRIDE: bar.exe
