// RUN: %clang_cc1 -E %s -Wall -ffreestanding -verify
// RUN: %clang_cc1 -Eonly %s -Wall -ffreestanding -verify
// RUN: %clang -M -Wall -ffreestanding %s -Xclang -verify
// RUN: %clang -E -frewrite-includes %s -Wall -ffreestanding -Xclang -verify
// RUN: %clang -E -dD -dM %s -Wall -ffreestanding -Xclang -verify
// expected-no-diagnostics

#pragma GCC visibility push (default)
#pragma weak
#pragma this_pragma_does_not_exist
