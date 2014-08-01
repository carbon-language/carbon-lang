// RUN: %clang -fmodules -fno-modules -### %s 2>&1 | FileCheck -check-prefix=CHECK-NO-MODULES %s
// CHECK-NO-MODULES-NOT: -fmodules

// RUN: %clang -fmodules -fno-modules -fmodules -### %s 2>&1 | FileCheck -check-prefix=CHECK-HAS-MODULES %s
// CHECK-HAS-MODULES: -fmodules

// RUN: %clang -fbuild-session-file=doesntexist -### %s 2>&1 | FileCheck -check-prefix=NOFILE %s
// NOFILE: no such file or directory: 'doesntexist'

// REQUIRES: shell
// RUN: touch -m -a -t 201008011501 %t.build-session-file
// RUN: %clang -fbuild-session-file=%t.build-session-file -### %s 2>&1 | FileCheck -check-prefix=TIMESTAMP_ONLY %s

// RUN: %clang -fbuild-session-timestamp=1280703457 -### %s 2>&1 | FileCheck -check-prefix=TIMESTAMP_ONLY %s
// TIMESTAMP_ONLY: -fbuild-session-timestamp=128

// RUN: %clang -fbuild-session-file=%t.build-session-file -fbuild-session-timestamp=123 -### %s 2>&1 | FileCheck -check-prefix=CONFLICT %s
// CONFLICT: error: invalid argument '-fbuild-session-file={{.*}}.build-session-file' not allowed with '-fbuild-session-timestamp'

// RUN: %clang -fbuild-session-timestamp=123 -fmodules-validate-once-per-build-session -### %s 2>&1 | FileCheck -check-prefix=MODULES_VALIDATE_ONCE %s
// MODULES_VALIDATE_ONCE: -fbuild-session-timestamp=123
// MODULES_VALIDATE_ONCE: -fmodules-validate-once-per-build-session

// RUN: %clang -fbuild-session-file=%t.build-session-file -fmodules-validate-once-per-build-session -### %s 2>&1 | FileCheck -check-prefix=MODULES_VALIDATE_ONCE_FILE %s
// MODULES_VALIDATE_ONCE_FILE: -fbuild-session-timestamp=128
// MODULES_VALIDATE_ONCE_FILE: -fmodules-validate-once-per-build-session

// RUN: %clang -fmodules-validate-once-per-build-session -### %s 2>&1 | FileCheck -check-prefix=MODULES_VALIDATE_ONCE_ERR %s
// MODULES_VALIDATE_ONCE_ERR: option '-fmodules-validate-once-per-build-session' requires '-fbuild-session-timestamp=<seconds since Epoch>' or '-fbuild-session-file=<file>'

// RUN: %clang -### %s 2>&1 | FileCheck -check-prefix=MODULES_VALIDATE_SYSTEM_HEADERS_DEFAULT %s
// MODULES_VALIDATE_SYSTEM_HEADERS_DEFAULT-NOT: -fmodules-validate-system-headers

// RUN: %clang -fmodules-validate-system-headers -### %s 2>&1 | FileCheck -check-prefix=MODULES_VALIDATE_SYSTEM_HEADERS %s
// MODULES_VALIDATE_SYSTEM_HEADERS: -fmodules-validate-system-headers
