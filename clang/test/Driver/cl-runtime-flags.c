// Don't attempt slash switches on msys bash.
// REQUIRES: shell-preserves-root

// Note: %s must be preceded by --, otherwise it may be interpreted as a
// command-line option, e.g. on Mac where %s is commonly under /Users.

// First check that regular clang doesn't do any of this stuff.
// RUN: %clang -### %s 2>&1 | FileCheck -check-prefix=CHECK-CLANG %s
// CHECK-CLANG-NOT: "-D_DEBUG"
// CHECK-CLANG-NOT: "-D_MT"
// CHECK-CLANG-NOT: "-D_DLL"
// CHECK-CLANG-NOT: --dependent-lib

// RUN: %clang_cl -### -- %s 2>&1 | FileCheck -check-prefix=CHECK-MT %s
// RUN: %clang_cl -### /MT -- %s 2>&1 | FileCheck -check-prefix=CHECK-MT %s
// CHECK-MT-NOT: "-D_DEBUG"
// CHECK-MT: "-D_MT"
// CHECK-MT-NOT: "-D_DLL"
// CHECK-MT: "--dependent-lib=libcmt"
// CHECK-MT: "--dependent-lib=oldnames"

// RUN: %clang_cl -### /MTd -- %s 2>&1 | FileCheck -check-prefix=CHECK-MTd %s
// CHECK-MTd: "-D_DEBUG"
// CHECK-MTd: "-D_MT"
// CHECK-MTd-NOT: "-D_DLL"
// CHECK-MTd: "--dependent-lib=libcmtd"
// CHECK-MTd: "--dependent-lib=oldnames"

// RUN: %clang_cl -### /MD -- %s 2>&1 | FileCheck -check-prefix=CHECK-MD %s
// CHECK-MD-NOT: "-D_DEBUG"
// CHECK-MD: "-D_MT"
// CHECK-MD: "-D_DLL"
// CHECK-MD: "--dependent-lib=msvcrt"
// CHECK-MD: "--dependent-lib=oldnames"

// RUN: %clang_cl -### /MDd -- %s 2>&1 | FileCheck -check-prefix=CHECK-MDd %s
// CHECK-MDd: "-D_DEBUG"
// CHECK-MDd: "-D_MT"
// CHECK-MDd: "-D_DLL"
// CHECK-MDd: "--dependent-lib=msvcrtd"
// CHECK-MDd: "--dependent-lib=oldnames"
