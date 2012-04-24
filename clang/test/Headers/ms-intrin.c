// RUN: %clang -fsyntax-only -ffreestanding -target i686-pc-win32 %s
// RUN: %clangxx -fsyntax-only -ffreestanding -target i686-pc-win32 -x c++ %s

#include <intrin.h>

// Ensure we're compiling in MS-compat mode.
__declspec(naked) void f();

// Ensure that we got various fundamental headers.
// FIXME: We could probably do more extensive testing here.
size_t test = 0;
