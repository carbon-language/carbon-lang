// RUN: %clang -fmodules %s -### 2>&1 | FileCheck -check-prefix CHECK-NO_MODULE_CACHE %s
// RUN: %clang -fmodules -fmodules-cache-path=blarg %s -### 2>&1 | FileCheck -check-prefix CHECK-WITH_MODULE_CACHE %s

// CHECK-NO_MODULE_CACHE: {{clang.*"-fmodules-cache-path=.*ModuleCache"}}

// CHECK-WITH_MODULE_CACHE: {{clang.*"-fmodules-cache-path=blarg"}}
