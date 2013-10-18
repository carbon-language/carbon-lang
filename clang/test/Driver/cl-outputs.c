// Don't attempt slash switches on msys bash.
// REQUIRES: shell-preserves-root

// Note: %s must be preceded by --, otherwise it may be interpreted as a
// command-line option, e.g. on Mac where %s is commonly under /Users.

// RUN: %clang_cl /c -### -- %s 2>&1 | FileCheck -check-prefix=DEFAULT %s
// DEFAULT: "-o" "cl-outputs.obj"

// RUN: %clang_cl /Foa -### -- %s 2>&1 | FileCheck -check-prefix=FoNAME %s
// FoNAME:  "-o" "a.obj"

// RUN: %clang_cl /Foa.ext /Fob.ext -### -- %s 2>&1 | FileCheck -check-prefix=FoNAMEEXT %s
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

// RUN: %clang_cl /LD -### -- %s 2>&1 | FileCheck -check-prefix=DEFAULTDLL %s
// RUN: %clang_cl /LDd -### -- %s 2>&1 | FileCheck -check-prefix=DEFAULTDLL %s
// DEFAULTDLL: "-out:cl-outputs.dll"
// DEFAULTDLL: "-implib:cl-outputs.lib"

// RUN: %clang_cl /Fefoo -### -- %s 2>&1 | FileCheck -check-prefix=FeNOEXT %s
// FeNOEXT: "-out:foo.exe"

// RUN: %clang_cl /Fefoo /LD -### -- %s 2>&1 | FileCheck -check-prefix=FeNOEXTDLL %s
// RUN: %clang_cl /Fefoo /LDd -### -- %s 2>&1 | FileCheck -check-prefix=FeNOEXTDLL %s
// FeNOEXTDLL: "-out:foo.dll"
// FeNOEXTDLL: "-implib:foo.lib"

// RUN: %clang_cl /Fefoo.ext -### -- %s 2>&1 | FileCheck -check-prefix=FeEXT %s
// FeEXT: "-out:foo.ext"

// RUN: %clang_cl /LD /Fefoo.ext -### -- %s 2>&1 | FileCheck -check-prefix=FeEXTDLL %s
// RUN: %clang_cl /LDd /Fefoo.ext -### -- %s 2>&1 | FileCheck -check-prefix=FeEXTDLL %s
// FeEXTDLL: "-out:foo.ext"
// FeEXTDLL: "-implib:foo.lib"

// RUN: %clang_cl /Fefoo.dir/ -### -- %s 2>&1 | FileCheck -check-prefix=FeDIR %s
// FeDIR: "-out:foo.dir{{[/\\]+}}cl-outputs.exe"

// RUN: %clang_cl /LD /Fefoo.dir/ -### -- %s 2>&1 | FileCheck -check-prefix=FeDIRDLL %s
// RUN: %clang_cl /LDd /Fefoo.dir/ -### -- %s 2>&1 | FileCheck -check-prefix=FeDIRDLL %s
// FeDIRDLL: "-out:foo.dir{{[/\\]+}}cl-outputs.dll"
// FeDIRDLL: "-implib:foo.dir{{[/\\]+}}cl-outputs.lib"

// RUN: %clang_cl /Fefoo.dir/a -### -- %s 2>&1 | FileCheck -check-prefix=FeDIRNAME %s
// FeDIRNAME: "-out:foo.dir{{[/\\]+}}a.exe"

// RUN: %clang_cl /LD /Fefoo.dir/a -### -- %s 2>&1 | FileCheck -check-prefix=FeDIRNAMEDLL %s
// RUN: %clang_cl /LDd /Fefoo.dir/a -### -- %s 2>&1 | FileCheck -check-prefix=FeDIRNAMEDLL %s
// FeDIRNAMEDLL: "-out:foo.dir{{[/\\]+}}a.dll"
// FeDIRNAMEDLL: "-implib:foo.dir{{[/\\]+}}a.lib"

// RUN: %clang_cl /Fefoo.dir/a.ext -### -- %s 2>&1 | FileCheck -check-prefix=FeDIRNAMEEXT %s
// FeDIRNAMEEXT: "-out:foo.dir{{[/\\]+}}a.ext"

// RUN: %clang_cl /LD /Fefoo.dir/a.ext -### -- %s 2>&1 | FileCheck -check-prefix=FeDIRNAMEEXTDLL %s
// RUN: %clang_cl /LDd /Fefoo.dir/a.ext -### -- %s 2>&1 | FileCheck -check-prefix=FeDIRNAMEEXTDLL %s
// FeDIRNAMEEXTDLL: "-out:foo.dir{{[/\\]+}}a.ext"
// FeDIRNAMEEXTDLL: "-implib:foo.dir{{[/\\]+}}a.lib"

// RUN: %clang_cl /Fe -### 2>&1 | FileCheck -check-prefix=FeMISSINGARG %s
// FeMISSINGARG: error: argument to '/Fe' is missing (expected 1 value)

// RUN: %clang_cl /Fefoo /Febar -### -- %s 2>&1 | FileCheck -check-prefix=FeOVERRIDE %s
// FeOVERRIDE: "-out:bar.exe"


// RUN: %clang_cl /FA -### -- %s 2>&1 | FileCheck -check-prefix=FA %s
// FA: "-o" "cl-outputs.asm"
// RUN: %clang_cl /FA /Fafoo -### -- %s 2>&1 | FileCheck -check-prefix=FaNAME %s
// RUN: %clang_cl /Fafoo -### -- %s 2>&1 | FileCheck -check-prefix=FaNAME %s
// FaNAME:  "-o" "foo.asm"
// RUN: %clang_cl /FA /Faa.ext /Fab.ext -### -- %s 2>&1 | FileCheck -check-prefix=FaNAMEEXT %s
// FaNAMEEXT:  "-o" "b.ext"
// RUN: %clang_cl /FA /Fafoo.dir/ -### -- %s 2>&1 | FileCheck -check-prefix=FaDIR %s
// FaDIR:  "-o" "foo.dir{{[/\\]+}}cl-outputs.asm"
// RUN: %clang_cl /FA /Fafoo.dir/a -### -- %s 2>&1 | FileCheck -check-prefix=FaDIRNAME %s
// FaDIRNAME:  "-o" "foo.dir{{[/\\]+}}a.asm"
// RUN: %clang_cl /FA /Fafoo.dir/a.ext -### -- %s 2>&1 | FileCheck -check-prefix=FaDIRNAMEEXT %s
// FaDIRNAMEEXT:  "-o" "foo.dir{{[/\\]+}}a.ext"
// RUN: %clang_cl /Faa.asm -### -- %s %s 2>&1 | FileCheck -check-prefix=FaMULTIPLESOURCE %s
// FaMULTIPLESOURCE: error: cannot specify '/Faa.asm' when compiling multiple source files
