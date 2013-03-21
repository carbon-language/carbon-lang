// RUN: %clang %s -fsyntax-only -### 2> %t.log
// RUN: FileCheck %s --check-prefix=CHECK-DEFAULT < %t.log

// CHECK-DEFAULT: "-resource-dir" "{{.+}}/../lib/clang/{{.+}}"

// RUN: %clang %s -fsyntax-only -ccc-install-dir /my/install/dir -### 2> %t.log
// RUN: FileCheck %s --check-prefix=CHECK-INSTALL-DIR < %t.log
// CHECK-INSTALL-DIR: "-resource-dir" "/my/install/dir{{[\\/]+}}..{{[\\/]+}}lib{{[\\/]+}}clang{{[\\/]+.+}}"

void foo(void) {}
