// Don't attempt slash switches on msys bash.
// REQUIRES: shell-preserves-root

// Note: %s must be preceded by --, otherwise it may be interpreted as a
// command-line option, e.g. on Mac where %s is commonly under /Users.

// RUN: %clang_cl /Foa -### -- %s 2>&1 | FileCheck -check-prefix=CHECK-NAME %s
// CHECK-NAME:  "-o" "a.obj"

// RUN: %clang_cl /Foa.ext /Fob.ext -### -- %s 2>&1 | FileCheck -check-prefix=CHECK-NAMEEXT %s
// CHECK-NAMEEXT:  warning: overriding '/Foa.ext' option with '/Fob.ext'
// CHECK-NAMEEXT:  "-o" "b.ext"

// RUN: %clang_cl /Fofoo.dir/ -### -- %s 2>&1 | FileCheck -check-prefix=CHECK-DIR %s
// CHECK-DIR:  "-o" "foo.dir{{[/\\]+}}cl-Fo.obj"

// RUN: %clang_cl /Fofoo.dir/a -### -- %s 2>&1 | FileCheck -check-prefix=CHECK-DIRNAME %s
// CHECK-DIRNAME:  "-o" "foo.dir{{[/\\]+}}a.obj"

// RUN: %clang_cl /Fofoo.dir/a.ext -### -- %s 2>&1 | FileCheck -check-prefix=CHECK-DIRNAMEEXT %s
// CHECK-DIRNAMEEXT:  "-o" "foo.dir{{[/\\]+}}a.ext"

// RUN: %clang_cl /Fo.. -### -- %s 2>&1 | FileCheck -check-prefix=CHECK-CRAZY %s
// CHECK-CRAZY:  "-o" "...obj"


// RUN: %clang_cl /Fo -### 2>&1 | FileCheck -check-prefix=CHECK-MISSINGARG %s
// CHECK-MISSINGARG: error: argument to '/Fo' is missing (expected 1 value)

// RUN: %clang_cl /Foa.obj -### -- %s %s 2>&1 | FileCheck -check-prefix=CHECK-MULTIPLESOURCEERROR %s
// CHECK-MULTIPLESOURCEERROR: error: cannot specify '/Foa.obj' when compiling multiple source files

// RUN: %clang_cl /Fomydir/ -### -- %s %s 2>&1 | FileCheck -check-prefix=CHECK-MULTIPLESOURCEOK %s
// CHECK-MULTIPLESOURCEOK: "-o" "mydir{{[/\\]+}}cl-Fo.obj"
