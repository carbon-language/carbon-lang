// RUN: %clang_cc1 -emit-llvm -g -triple x86_64-apple-darwin %s -o - | FileCheck %s

template<typename _CharT>
struct basic_string {

  basic_string&
  assign(const _CharT* __s, const basic_string<_CharT> &x)
  {
    return *this;
  }
};

void foo (const char *c) {
  basic_string<char> str;
  str.assign(c, str);
}

// CHECK: [[P:.*]] = metadata !{i32 {{.*}}, metadata [[CON:.*]]} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [from ]
// CHECK: [[CON]] = metadata !{i32 {{.*}}, metadata [[CH:.*]]} ; [ DW_TAG_const_type ] [line 0, size 0, align 0, offset 0] [from char]
// CHECK: [[CH]] = metadata !{i32 {{.*}}, metadata !"char", {{.*}}} ; [ DW_TAG_base_type ] [char] [line 0, size 8, align 8, offset 0, enc DW_ATE_signed_char]
// CHECK: metadata !{i32 {{.*}}, metadata !"_ZN12basic_stringIcE6assignEPKcRKS0_", metadata ![[FILE:.*]], i32 7, metadata [[TYPE:.*]], i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 false, %struct.basic_string* (%struct.basic_string*, i8*, %struct.basic_string*)* @_ZN12basic_stringIcE6assignEPKcRKS0_, null, metadata !{{.*}}, metadata !{{.*}}, i32 8} ; [ DW_TAG_subprogram ] [line 7] [def] [scope 8] [assign]
// CHECK: [[TYPE]] = metadata !{i32 {{.*}}, null, metadata [[ARGS:.*]], i32 0, i32 0}
// CHECK: [[ARGS]] = metadata !{metadata !{{.*}}, metadata !{{.*}}, metadata [[P]], metadata [[R:.*]]}
// CHECK: [[BS:.*]] = metadata !{i32 {{.*}}, null, metadata !"basic_string<char>", metadata ![[FILE]], i32 4, i64 8, i64 8, i32 0, i32 0, null, metadata !{{.*}}, i32 0, null, metadata !{{.*}}} ; [ DW_TAG_structure_type ] [basic_string<char>] [line 4, size 8, align 8, offset 0] [from ]
// CHECK: [[R]] = metadata !{i32 {{.*}}, null, null, null, i32 0, i64 0, i64 0, i64 0, i32 0, metadata [[CON2:.*]]} ; [ DW_TAG_reference_type ] [line 0, size 0, align 0, offset 0] [from ]
// CHECK: [[CON2]] = metadata !{i32 {{.*}}, metadata [[BS]]} ; [ DW_TAG_const_type ] [line 0, size 0, align 0, offset 0] [from basic_string<char>]
