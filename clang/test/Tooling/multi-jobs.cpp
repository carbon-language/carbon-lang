// RUN: not clang-check "%s" -- -no-integrated-as -c 2>&1 | FileCheck %s

// CHECK: C++ requires
invalid;

// MSVC targeted drivers (*-win32) are incapable of invoking external assembler.
// XFAIL: win32
