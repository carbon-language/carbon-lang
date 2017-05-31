// Note: %s must be preceded by --, otherwise it may be interpreted as a
// command-line option, e.g. on Mac where %s is commonly under /Users.

// RUN: %clang_cl --target=i686-windows-msvc /Gd -### -- %s 2>&1 | FileCheck --check-prefix=CDECL %s
// CDECL: -fdefault-calling-convention=cdecl

// RUN: %clang_cl --target=i686-windows-msvc /Gr -### -- %s 2>&1 | FileCheck --check-prefix=FASTCALL %s
// FASTCALL: -fdefault-calling-convention=fastcall

// RUN: %clang_cl --target=i686-windows-msvc /Gz -### -- %s 2>&1 | FileCheck --check-prefix=STDCALL %s
// STDCALL: -fdefault-calling-convention=stdcall

// RUN: %clang_cl --target=i686-windows-msvc /Gv -### -- %s 2>&1 | FileCheck --check-prefix=VECTORCALL %s
// VECTORCALL: -fdefault-calling-convention=vectorcall

// Last one should win:

// RUN: %clang_cl --target=i686-windows-msvc /Gd /Gv -### -- %s 2>&1 | FileCheck --check-prefix=LASTWINS_VECTOR %s
// LASTWINS_VECTOR: -fdefault-calling-convention=vectorcall

// RUN: %clang_cl --target=i686-windows-msvc /Gv /Gd -### -- %s 2>&1 | FileCheck --check-prefix=LASTWINS_CDECL %s
// LASTWINS_CDECL: -fdefault-calling-convention=cdecl

// No fastcall or stdcall on x86_64:

// RUN: %clang_cl --target=x86_64-windows-msvc /Gr -### -- %s 2>&1 | FileCheck --check-prefix=UNSUPPORTED %s
// RUN: %clang_cl --target=x86_64-windows-msvc /Gz -### -- %s 2>&1 | FileCheck --check-prefix=UNSUPPORTED %s
// RUN: %clang_cl --target=thumbv7-windows-msvc /Gv -### -- %s 2>&1 | FileCheck --check-prefix=UNSUPPORTED %s

// UNSUPPORTED-NOT: error:
// UNSUPPORTED-NOT: warning:
// UNSUPPORTED-NOT: -fdefault-calling-convention=

