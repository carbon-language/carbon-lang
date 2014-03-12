// RUN: %clang -fmodules -fno-modules -### %s 2>&1 | FileCheck -check-prefix=CHECK-NO-MODULES %s
// CHECK-NO-MODULES-NOT: -fmodules

// RUN: %clang -fmodules -fno-modules -fmodules -### %s 2>&1 | FileCheck -check-prefix=CHECK-HAS-MODULES %s
// CHECK-HAS-MODULES: -fmodules

// RUN: %clang -fbuild-session-timestamp=123 -### %s 2>&1 | FileCheck -check-prefix=TIMESTAMP_ONLY %s
// TIMESTAMP_ONLY: -fbuild-session-timestamp=123

// RUN: %clang -fbuild-session-timestamp=123 -fmodules-validate-once-per-build-session -### %s 2>&1 | FileCheck -check-prefix=MODULES_VALIDATE_ONCE %s
// MODULES_VALIDATE_ONCE: -fbuild-session-timestamp=123
// MODULES_VALIDATE_ONCE: -fmodules-validate-once-per-build-session

// RUN: %clang -fmodules-validate-once-per-build-session -### %s 2>&1 | FileCheck -check-prefix=MODULES_VALIDATE_ONCE_ERR %s
// MODULES_VALIDATE_ONCE_ERR: option '-fmodules-validate-once-per-build-session' requires '-fbuild-session-timestamp=<seconds since Epoch>'

// RUN: %clang -### %s 2>&1 | FileCheck -check-prefix=MODULES_VALIDATE_SYSTEM_HEADERS_DEFAULT %s
// MODULES_VALIDATE_SYSTEM_HEADERS_DEFAULT-NOT: -fmodules-validate-system-headers

// RUN: %clang -fmodules-validate-system-headers -### %s 2>&1 | FileCheck -check-prefix=MODULES_VALIDATE_SYSTEM_HEADERS %s
// MODULES_VALIDATE_SYSTEM_HEADERS: -fmodules-validate-system-headers
