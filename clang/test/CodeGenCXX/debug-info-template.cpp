// RUN: %clang_cc1 -emit-llvm -g %s -o - | FileCheck %s

// CHECK: metadata [[TCI:![0-9]*]], i32 0, i32 1, %class.TC* @tci, null} ; [ DW_TAG_variable ] [tci]
// CHECK: [[TC:![0-9]*]] = {{.*}}, metadata [[TCARGS:![0-9]*]]} ; [ DW_TAG_class_type ] [TC<int, 2>]
// CHECK: [[TCARGS]] = metadata !{metadata [[TCARG1:![0-9]*]], metadata [[TCARG2:![0-9]*]]}
//
// We seem to be missing file/line/col info on template value parameters -
// metadata supports it but it's not populated.
//
// CHECK: [[TCARG1]] = {{.*}}metadata !"T", metadata [[INT:![0-9]*]], {{.*}} ; [ DW_TAG_template_type_parameter ]
// CHECK: [[INT]] = {{.*}} ; [ DW_TAG_base_type ] [int]
// CHECK: [[TCARG2]] = {{.*}}metadata !"", metadata [[UINT:![0-9]*]], i64 2, {{.*}} ; [ DW_TAG_template_value_parameter ]
// CHECK: [[UINT]] = {{.*}} ; [ DW_TAG_base_type ] [unsigned int]

template<typename T, unsigned>
class TC {
};

TC<int, 2> tci;
