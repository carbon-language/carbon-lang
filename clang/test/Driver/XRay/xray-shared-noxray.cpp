// RUN: %clangxx -shared -fPIC -o /dev/null -v -fxray-instrument %s 2>&1 | \
// RUN:     FileCheck %s --check-prefix=SHARED
// RUN: %clangxx -static -o /dev/null -v -fxray-instrument %s 2>&1 -DMAIN | \
// RUN:     FileCheck %s --check-prefix=STATIC
// RUN: %clangxx -static -fPIE -o /dev/null -v -fxray-instrument %s 2>&1 \
// RUN:     -DMAIN | FileCheck %s --check-prefix=STATIC
//
// SHARED-NOT: {{clang_rt\.xray-}}
// STATIC: {{clang_rt\.xray-}}
int foo() { return 42; }

#ifdef MAIN
int main() { return foo(); }
#endif
