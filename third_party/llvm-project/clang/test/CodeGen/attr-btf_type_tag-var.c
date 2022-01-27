// RUN: %clang_cc1 -triple %itanium_abi_triple -debug-info-kind=limited -S -emit-llvm -o - %s | FileCheck %s

#define __tag1 __attribute__((btf_type_tag("tag1")))
#define __tag2 __attribute__((btf_type_tag("tag2")))
#define __tag3 __attribute__((btf_type_tag("tag3")))
#define __tag4 __attribute__((btf_type_tag("tag4")))
#define __tag5 __attribute__((btf_type_tag("tag5")))
#define __tag6 __attribute__((btf_type_tag("tag6")))

const int __tag1 __tag2 volatile * const __tag3  __tag4  volatile * __tag5  __tag6 const volatile * g;

// CHECK:  distinct !DIGlobalVariable(name: "g", scope: ![[#]], file: ![[#]], line: [[#]], type: ![[L6:[0-9]+]]
// CHECK:  ![[L6]] = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: ![[L7:[0-9]+]], size: [[#]], annotations: ![[L22:[0-9]+]]
// CHECK:  ![[L7]] = !DIDerivedType(tag: DW_TAG_const_type, baseType: ![[L8:[0-9]+]]
// CHECK:  ![[L8]] = !DIDerivedType(tag: DW_TAG_volatile_type, baseType: ![[L9:[0-9]+]]
// CHECK:  ![[L9]] = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: ![[L10:[0-9]+]], size: [[#]], annotations: ![[L19:[0-9]+]]
// CHECK:  ![[L10]] = !DIDerivedType(tag: DW_TAG_const_type, baseType: ![[L11:[0-9]+]]
// CHECK:  ![[L11]] = !DIDerivedType(tag: DW_TAG_volatile_type, baseType: ![[L12:[0-9]+]]
// CHECK:  ![[L12]] = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: ![[L13:[0-9]+]], size: [[#]], annotations: ![[L16:[0-9]+]]
// CHECK:  ![[L13]] = !DIDerivedType(tag: DW_TAG_const_type, baseType: ![[L14:[0-9]+]]
// CHECK:  ![[L14]] = !DIDerivedType(tag: DW_TAG_volatile_type, baseType: ![[L15:[0-9]+]]
// CHECK:  ![[L15]] = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed
// CHECK:  ![[L16]] = !{![[L17:[0-9]+]], ![[L18:[0-9]+]]}
// CHECK:  ![[L17]] = !{!"btf_type_tag", !"tag1"}
// CHECK:  ![[L18]] = !{!"btf_type_tag", !"tag2"}
// CHECK:  ![[L19]] = !{![[L20:[0-9]+]], ![[L21:[0-9]+]]}
// CHECK:  ![[L20]] = !{!"btf_type_tag", !"tag3"}
// CHECK:  ![[L21]] = !{!"btf_type_tag", !"tag4"}
// CHECK:  ![[L22]] = !{![[L23:[0-9]+]], ![[L24:[0-9]+]]}
// CHECK:  ![[L23]] = !{!"btf_type_tag", !"tag5"}
// CHECK:  ![[L24]] = !{!"btf_type_tag", !"tag6"}
