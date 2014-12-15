// RUN: %clang_cc1 -emit-llvm -fms-extensions -triple=x86_64-pc-win32 -g %s -o - -std=c++11 | FileCheck %s
// RUN: %clang_cc1 -emit-llvm -fms-extensions -triple=x86_64-unknown-unknown -g %s -o - -std=c++11 2>&1 | FileCheck %s --check-prefix=CHECK-ITANIUM

// CHECK: [[TGIARGS:![0-9]*]], null} ; [ DW_TAG_structure_type ] [tmpl_guid<&__uuidof(uuid)>]
// CHECK: [[TGIARGS]] = !{[[TGIARG1:![0-9]*]]}
// CHECK: [[TGIARG1]] = !{!"0x30\00\00{{.*}}", {{[^,]+}}, [[CONST_GUID_PTR:![0-9]*]], { i32, i16, i16, [8 x i8] }* @_GUID_12345678_1234_1234_1234_1234567890ab, {{.*}} ; [ DW_TAG_template_value_parameter ]
// CHECK: [[CONST_GUID_PTR]] = {{.*}}, [[CONST_GUID:![0-9]*]]} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [from ]
// CHECK: [[CONST_GUID]] = {{.*}}, [[GUID:![0-9]*]]} ; [ DW_TAG_const_type ] [line 0, size 0, align 0, offset 0] [from _GUID]
// CHECK: [[GUID]] = {{.*}} ; [ DW_TAG_structure_type ] [_GUID]

// CHECK: [[TGI2ARGS:![0-9]*]], null} ; [ DW_TAG_structure_type ] [tmpl_guid2<__uuidof(uuid)>]
// CHECK: [[TGI2ARGS]] = !{[[TGI2ARG1:![0-9]*]]}
// CHECK: [[TGI2ARG1]] = !{!"0x30\00\00{{.*}}", {{[^,]+}}, [[CONST_GUID_REF:![0-9]*]], { i32, i16, i16, [8 x i8] }* @_GUID_12345678_1234_1234_1234_1234567890ab, {{.*}} ; [ DW_TAG_template_value_parameter ]
// CHECK: [[CONST_GUID_REF]] = {{.*}}, [[CONST_GUID:![0-9]*]]} ; [ DW_TAG_reference_type ] [line 0, size 0, align 0, offset 0] [from ]

// CHECK-ITANIUM: !"_ZTS9tmpl_guidIXadu8__uuidoft4uuidEE"} ; [ DW_TAG_structure_type ] [tmpl_guid<&__uuidof(uuid)>]
// CHECK-ITANIUM: !"_ZTS10tmpl_guid2IXu8__uuidoft4uuidEE"} ; [ DW_TAG_structure_type ] [tmpl_guid2<__uuidof(uuid)>]

struct _GUID;
template <const _GUID *>
struct tmpl_guid {
};

struct __declspec(uuid("{12345678-1234-1234-1234-1234567890ab}")) uuid;
tmpl_guid<&__uuidof(uuid)> tgi;

template <const _GUID &>
struct tmpl_guid2 {};
tmpl_guid2<__uuidof(uuid)> tgi2;
