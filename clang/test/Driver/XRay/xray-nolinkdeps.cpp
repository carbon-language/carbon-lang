// RUN: %clang -v -o /dev/null -fxray-instrument -fnoxray-link-deps %s -### \
// RUN:     2>&1 | FileCheck --check-prefix DISABLE %s
// RUN: %clang -v -o /dev/null -fxray-instrument -fxray-link-deps %s -### \
// RUN:     2>&1 | FileCheck --check-prefix ENABLE %s
// ENABLE: clang_rt.xray
// DISABLE-NOT: clang_rt.xray
