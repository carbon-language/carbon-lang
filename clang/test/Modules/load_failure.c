#ifdef NONEXISTENT
__import_module__ load_nonexistent;
#endif

#ifdef FAILURE
__import_module__ load_failure;
#endif

// RUN: rm -rf %t
// RUN: %clang_cc1 -x c++ -fmodule-cache-path %t -fdisable-module-hash -emit-module -fmodule-name=load_failure %S/Inputs/module.map
// RUN: %clang_cc1 -fmodule-cache-path %t -fdisable-module-hash %s -DNONEXISTENT 2>&1 | FileCheck -check-prefix=CHECK-NONEXISTENT %s
// CHECK-NONEXISTENT: load_failure.c:2:19: fatal error: module 'load_nonexistent' not found

// RUN: not %clang_cc1 -fmodule-cache-path %t -fdisable-module-hash %s -DFAILURE 2> %t.out
// RUN: FileCheck -check-prefix=CHECK-FAILURE %s < %t.out

// FIXME: Clean up diagnostic text below and give it a location
// CHECK-FAILURE: error: C99 was disabled in PCH file but is currently enabled


