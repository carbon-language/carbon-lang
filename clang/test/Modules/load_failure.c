#ifdef NONEXISTENT
__import_module__ load_nonexistent;
#endif

#ifdef FAILURE
__import_module__ load_failure;
#endif

// RUN: %clang_cc1 -x c++ -emit-module -o %T/load_failure.pcm %S/Inputs/load_failure.h
// RUN: %clang_cc1 -I %T %s -DNONEXISTENT 2>&1 | FileCheck -check-prefix=CHECK-NONEXISTENT %s
// CHECK-NONEXISTENT: load_failure.c:2:19: fatal error: module 'load_nonexistent' not found

// RUN: not %clang_cc1 -I %T %s -DFAILURE 2> %t
// RUN: FileCheck -check-prefix=CHECK-FAILURE %s < %t

// FIXME: Clean up diagnostic text below and give it a location
// CHECK-FAILURE: error: C99 support was disabled in PCH file but is currently enabled


