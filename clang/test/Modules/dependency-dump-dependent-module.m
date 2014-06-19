// When a module depends on another, check that we dump the dependency header
// files for both.

// RUN: rm -rf %t
// RUN: %clang_cc1 -fmodules -fmodules-cache-path=%t/cache -module-dependency-dir %t/vfs -F %S/Inputs -I %S/Inputs -verify %s
// expected-no-diagnostics

// RUN: FileCheck %s -check-prefix=VFS < %t/vfs/vfs.yaml
// VFS: 'name': "AlsoDependsOnModule.h"
// VFS: 'name': "SubFramework.h"
// VFS: 'name': "Treasure.h"
// VFS: 'name': "Module.h"
// VFS: 'name': "Sub.h"
// VFS: 'name': "Sub2.h"

// TODO: We need shell to use find here. Is there a simpler way?
// REQUIRES: shell

// RUN: find %t/vfs -type f | FileCheck %s -check-prefix=DUMP
// DUMP: AlsoDependsOnModule.framework/Headers/AlsoDependsOnModule.h
// DUMP: Module.framework/Frameworks/SubFramework.framework/Headers/SubFramework.h
// DUMP: Module.framework/Headers/Buried/Treasure.h
// DUMP: Module.framework/Headers/Module.h
// DUMP: Module.framework/Headers/Sub.h
// DUMP: Module.framework/Headers/Sub2.h

@import AlsoDependsOnModule;
