// Don't attempt slash switches on msys bash.
// REQUIRES: shell-preserves-root

// Note: %s must be preceded by --, otherwise it may be interpreted as a
// command-line option, e.g. on Mac where %s is commonly under /Users.

// RUN: %clang_cl /fallback /Dfoo=bar /Ubaz /Ifoo /O0 /Ox /GR /GR- /Gy /Gy- \
// RUN:   /Gw /Gw- /LD /LDd /MD /MDd /MTd /MT /FImyheader.h /Zi \
// RUN:   -### -- %s 2>&1 \
// RUN:   | FileCheck %s
// CHECK: "-fdiagnostics-format" "msvc-fallback"
// CHECK: ||
// CHECK: cl.exe
// CHECK: "/nologo"
// CHECK: "/c"
// CHECK: "/W0"
// CHECK: "-D" "foo=bar"
// CHECK: "-U" "baz"
// CHECK: "-I" "foo"
// CHECK: "/Ox"
// CHECK: "/GR-"
// CHECK: "/Gy-"
// CHECK: "/Gw-"
// CHECK: "/Z7"
// CHECK: "/FImyheader.h"
// CHECK: "/LD"
// CHECK: "/LDd"
// CHECK: "/MT"
// CHECK: "/Tc" "{{.*cl-fallback.c}}"
// CHECK: "/Fo{{.*cl-fallback.*.obj}}"

// RUN: %clang_cl /fallback /Od -### -- %s 2>&1 | FileCheck -check-prefix=O0 %s
// O0: cl.exe
// O0: "/Od"
// RUN: %clang_cl /fallback /O1 -### -- %s 2>&1 | FileCheck -check-prefix=O1 %s
// O1: cl.exe
// O1: "-O1"
// RUN: %clang_cl /fallback /O2 -### -- %s 2>&1 | FileCheck -check-prefix=O2 %s
// O2: cl.exe
// O2: "-O2"
// RUN: %clang_cl /fallback /Os -### -- %s 2>&1 | FileCheck -check-prefix=Os %s
// Os: cl.exe
// Os: "-Os"
// RUN: %clang_cl /fallback /Ox -### -- %s 2>&1 | FileCheck -check-prefix=Ox %s
// Ox: cl.exe
// Ox: "/Ox"

// Only fall back when actually compiling, not for e.g. /P (preprocess).
// RUN: %clang_cl /fallback /P -### -- %s 2>&1 | FileCheck -check-prefix=P %s
// P-NOT: ||
// P-NOT: "cl.exe"

// RUN: not %clang_cl /fallback /c -- %s 2>&1 | \
// RUN:     FileCheck -check-prefix=ErrWarn %s
// ErrWarn: warning: falling back to {{.*}}cl.exe

// Don't attempt to run clang -cc1 with /fallback and /GR.  It isn't ready yet.
// RUN: %clang_cl /fallback /c /GR -### -- %s 2>&1 | \
// RUN:     FileCheck -check-prefix=RTTI %s
// RTTI: warning: cannot compile RTTI yet, falling back to {{.*}}cl.exe
// RUN: %clang_cl /fallback /c /GR /GR- -### -- %s 2>&1 | \
// RUN:     FileCheck -check-prefix=NO_RTTI %s
// NO_RTTI: "-cc1"
// NO_RTTI: ||
// NO_RTTI: cl.exe

// Don't fall back on non-C or C++ files.
// RUN: %clang_cl /fallback -### -- %S/Inputs/file.ll 2>&1 | FileCheck -check-prefix=LL %s
// LL: file.ll
// LL-NOT: ||
// LL-NOT: "cl.exe"


#error "This fails to compile."
