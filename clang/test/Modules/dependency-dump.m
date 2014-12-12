// Check that we can dump all of the headers a module depends on, and a VFS map
// for the same.

// RUN: rm -rf %t
// RUN: %clang_cc1 -fmodules -fmodules-cache-path=%t/cache -module-dependency-dir %t/vfs -F %S/Inputs -I %S/Inputs -verify %s
// expected-no-diagnostics

// RUN: FileCheck %s -check-prefix=VFS -input-file %t/vfs/vfs.yaml
// VFS-DAG: 'name': "SubFramework.h"
// VFS-DAG: 'name': "Treasure.h"
// VFS-DAG: 'name': "Module.h"
// VFS-DAG: 'name': "Sub.h"
// VFS-DAG: 'name': "Sub2.h"

@import Module;
