// RUN: %clang_cc1 -cl-std=CL2.0 -debug-info-kind=limited -dwarf-version=5 -emit-llvm -O0 -triple spir-unknown-unknown -o - %s | FileCheck %s
// RUN: %clang_cc1 -cl-std=CL2.0 -debug-info-kind=limited -dwarf-version=5 -emit-llvm -O0 -triple spir64-unknown-unknown -o - %s | FileCheck %s

// CHECK-DAG: ![[DWARF_ADDRESS_SPACE_GLOBAL:[0-9]+]] = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !{{[0-9]+}}, size: {{[0-9]+}}, dwarfAddressSpace: 1)
// CHECK-DAG: ![[DWARF_ADDRESS_SPACE_CONSTANT:[0-9]+]] = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !{{[0-9]+}}, size: {{[0-9]+}}, dwarfAddressSpace: 2)
// CHECK-DAG: ![[DWARF_ADDRESS_SPACE_LOCAL:[0-9]+]] = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !{{[0-9]+}}, size: {{[0-9]+}}, dwarfAddressSpace: 3)
// CHECK-DAG: ![[DWARF_ADDRESS_SPACE_PRIVATE:[0-9]+]] = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !{{[0-9]+}}, size: {{[0-9]+}}, dwarfAddressSpace: 0)
// CHECK-DAG: ![[DWARF_ADDRESS_SPACE_GENERIC:[0-9]+]] = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !{{[0-9]+}}, size: {{[0-9]+}}, dwarfAddressSpace: 4)

// CHECK-DAG: distinct !DIGlobalVariable(name: "FileVar0", scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: {{[0-9]+}}, type: ![[DWARF_ADDRESS_SPACE_GLOBAL]], isLocal: false, isDefinition: true)
global int *FileVar0;

// CHECK-DAG: distinct !DIGlobalVariable(name: "FileVar1", scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: {{[0-9]+}}, type: ![[DWARF_ADDRESS_SPACE_CONSTANT]], isLocal: false, isDefinition: true)
constant int *FileVar1;

// CHECK-DAG: distinct !DIGlobalVariable(name: "FileVar2", scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: {{[0-9]+}}, type: ![[DWARF_ADDRESS_SPACE_LOCAL]], isLocal: false, isDefinition: true)
local int *FileVar2;

// CHECK-DAG: distinct !DIGlobalVariable(name: "FileVar3", scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: {{[0-9]+}}, type: ![[DWARF_ADDRESS_SPACE_PRIVATE]], isLocal: false, isDefinition: true)
private int *FileVar3;

// CHECK-DAG: distinct !DIGlobalVariable(name: "FileVar4", scope: !{{[0-9]+}}, file: !{{[0-9]+}}, line: {{[0-9]+}}, type: ![[DWARF_ADDRESS_SPACE_GENERIC]], isLocal: false, isDefinition: true)
int *FileVar4;
