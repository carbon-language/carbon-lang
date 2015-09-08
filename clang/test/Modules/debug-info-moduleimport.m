// RUN: rm -rf %t
// RUN: %clang_cc1 -g -fmodules -DGREETING="Hello World" -UNDEBUG -fimplicit-module-maps -fmodules-cache-path=%t %s -I %S/Inputs -isysroot /tmp/.. -I %t -emit-llvm -o - | FileCheck %s

// CHECK: ![[CU:.*]] = distinct !DICompileUnit
@import DebugObjC;
// CHECK: !DIImportedEntity(tag: DW_TAG_imported_declaration, scope: ![[CU]], entity: ![[MODULE:.*]], line: 5)
// CHECK: ![[MODULE]] = !DIModule(scope: null, name: "DebugObjC", configMacros: "\22-DGREETING=Hello World\22 \22-UNDEBUG\22", includePath: "{{.*}}test{{.*}}Modules{{.*}}Inputs", isysroot: "/tmp/..")
