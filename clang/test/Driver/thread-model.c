// RUN: %clang -### -target arm-unknown-linux-gnu -c %s -mthread-model posix -v 2>&1 | FileCheck -check-prefix=CHECK-POSIX %s
// RUN: %clang -### -target arm-unknown-linux-gnu -c %s -mthread-model single -v 2>&1 | FileCheck -check-prefix=CHECK-SINGLE %s
// RUN: not %clang -target arm-unknown-linux-gnu -c %s -mthread-model silly -v 2>&1 | FileCheck -check-prefix=CHECK-INVALID %s
// CHECK-POSIX: "-mthread-model" "posix"
// CHECK-SINGLE: "-mthread-model" "single"
// CHECK-INVALID: error: invalid thread model 'silly' in '-mthread-model silly' for this target

// RUN: %clang -### -target arm-unknown-linux-gnu -c %s -v 2>&1 | FileCheck -check-prefix=CHECK-LINUX-POSIX %s
// RUN: %clang -### -target arm-unknown-linux-gnu -c %s -v -mthread-model single 2>&1 | FileCheck -check-prefix=CHECK-LINUX-SINGLE %s
// RUN: %clang -### -target arm-unknown-linux-gnu -c %s -v -mthread-model silly 2>&1 | FileCheck -check-prefix=CHECK-LINUX-INVALID %s
// CHECK-LINUX-POSIX: Thread model: posix
// CHECK-LINUX-POSIX: "-mthread-model" "posix"
// CHECK-LINUX-SINGLE: Thread model: single
// CHECK-LINUX-SINGLE: "-mthread-model" "single"
// CHECK-LINUX-INVALID-NOT: Thread model:

// RUN: %clang -### -target wasm32-unknown-linux-gnu -c %s -v 2>&1 | FileCheck -check-prefix=CHECK-WEBASSEMBLY-DEFAULT %s
// RUN: %clang -### -target wasm32-unknown-linux-gnu -c %s -v -mthread-model single 2>&1 | FileCheck -check-prefix=CHECK-WEBASSEMBLY-SINGLE %s
// RUN: %clang -### -target wasm32-unknown-linux-gnu -c %s -v -mthread-model posix 2>&1 | FileCheck -check-prefix=CHECK-WEBASSEMBLY-POSIX %s
// RUN: %clang -### -target wasm32-unknown-linux-gnu -c %s -v -mthread-model silly 2>&1 | FileCheck -check-prefix=CHECK-WEBASSEMBLY-INVALID %s
// RUN: %clang -### -target wasm64-unknown-linux-gnu -c %s -v 2>&1 | FileCheck -check-prefix=CHECK-WEBASSEMBLY-DEFAULT %s
// RUN: %clang -### -target wasm64-unknown-linux-gnu -c %s -v -mthread-model single 2>&1 | FileCheck -check-prefix=CHECK-WEBASSEMBLY-SINGLE %s
// RUN: %clang -### -target wasm64-unknown-linux-gnu -c %s -v -mthread-model posix 2>&1 | FileCheck -check-prefix=CHECK-WEBASSEMBLY-POSIX %s
// RUN: %clang -### -target wasm64-unknown-linux-gnu -c %s -v -mthread-model silly 2>&1 | FileCheck -check-prefix=CHECK-WEBASSEMBLY-INVALID %s
// CHECK-WEBASSEMBLY-DEFAULT: Thread model: posix
// CHECK-WEBASSEMBLY-DEFAULT: "-mthread-model" "posix"
// CHECK-WEBASSEMBLY-SINGLE: Thread model: single
// CHECK-WEBASSEMBLY-SINGLE: "-mthread-model" "single"
// CHECK-WEBASSEMBLY-POSIX: Thread model: posix
// CHECK-WEBASSEMBLY-POSIX: "-mthread-model" "posix"
// CHECK-WEBASSEMBLY-INVALID-NOT: Thread model:
