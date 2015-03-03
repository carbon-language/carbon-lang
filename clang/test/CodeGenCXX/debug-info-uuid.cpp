// RUN: %clang_cc1 -emit-llvm -fms-extensions -triple=x86_64-pc-win32 -g %s -o - -std=c++11 | FileCheck %s
// RUN: %clang_cc1 -emit-llvm -fms-extensions -triple=x86_64-unknown-unknown -g %s -o - -std=c++11 2>&1 | FileCheck %s --check-prefix=CHECK-ITANIUM

// CHECK: !MDCompositeType(tag: DW_TAG_structure_type, name: "tmpl_guid<&__uuidof(uuid)>"
// CHECK-SAME:             templateParams: [[TGIARGS:![0-9]*]]
// CHECK: [[TGIARGS]] = !{[[TGIARG1:![0-9]*]]}
// CHECK: [[TGIARG1]] = !MDTemplateValueParameter(
// CHECK-SAME:                                    type: [[CONST_GUID_PTR:![0-9]*]]
// CHECK-SAME:                                    value: { i32, i16, i16, [8 x i8] }* @_GUID_12345678_1234_1234_1234_1234567890ab
// CHECK: [[CONST_GUID_PTR]] = !MDDerivedType(tag: DW_TAG_pointer_type
// CHECK-SAME:                                baseType: [[CONST_GUID:![0-9]*]]
// CHECK-SAME:                                size: 64
// CHECK-SAME:                                align: 64
// CHECK: [[CONST_GUID]] = !MDDerivedType(tag: DW_TAG_const_type
// CHECK-SAME:                            baseType: [[GUID:![0-9]*]]
// CHECK: [[GUID]] = !MDCompositeType(tag: DW_TAG_structure_type, name: "_GUID"

// CHECK: !MDCompositeType(tag: DW_TAG_structure_type, name: "tmpl_guid2<__uuidof(uuid)>"
// CHECK-SAME:             templateParams: [[TGI2ARGS:![0-9]*]]
// CHECK: [[TGI2ARGS]] = !{[[TGI2ARG1:![0-9]*]]}
// CHECK: [[TGI2ARG1]] = !MDTemplateValueParameter(
// CHECK-SAME:                                     type: [[CONST_GUID_REF:![0-9]*]]
// CHECK-SAME:                                     value: { i32, i16, i16, [8 x i8] }* @_GUID_12345678_1234_1234_1234_1234567890ab
// CHECK: [[CONST_GUID_REF]] = !MDDerivedType(tag: DW_TAG_reference_type,
// CHECK-SAME:                                baseType: [[CONST_GUID:![0-9]*]]

// CHECK-ITANIUM: !MDCompositeType(tag: DW_TAG_structure_type, name: "tmpl_guid<&__uuidof(uuid)>"
// CHECK-ITANIUM-SAME:             identifier: "_ZTS9tmpl_guidIXadu8__uuidoft4uuidEE"
// CHECK-ITANIUM: !MDCompositeType(tag: DW_TAG_structure_type, name: "tmpl_guid2<__uuidof(uuid)>"
// CHECK-ITANIUM-SAME:             identifier: "_ZTS10tmpl_guid2IXu8__uuidoft4uuidEE"

struct _GUID;
template <const _GUID *>
struct tmpl_guid {
};

struct __declspec(uuid("{12345678-1234-1234-1234-1234567890ab}")) uuid;
tmpl_guid<&__uuidof(uuid)> tgi;

template <const _GUID &>
struct tmpl_guid2 {};
tmpl_guid2<__uuidof(uuid)> tgi2;
