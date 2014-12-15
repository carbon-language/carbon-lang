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

// CHECK: [[BS:.*]] = {{.*}} ; [ DW_TAG_structure_type ] [basic_string<char>] [line 4, size 8, align 8, offset 0] [def] [from ]
// CHECK: [[TYPE:![0-9]*]] = !{!"0x15\00{{.*}}"{{.*}}, [[ARGS:.*]], null, null, null} ; [ DW_TAG_subroutine_type ]
// CHECK: [[ARGS]] = !{!{{.*}}, !{{.*}}, [[P:![0-9]*]], [[R:.*]]}
// CHECK: [[P]] = {{.*}}, [[CON:![0-9]*]]} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [from ]
// CHECK: [[CON]] = {{.*}}, [[CH:![0-9]*]]} ; [ DW_TAG_const_type ] [line 0, size 0, align 0, offset 0] [from char]
// CHECK: [[CH]] = {{.*}} ; [ DW_TAG_base_type ] [char] [line 0, size 8, align 8, offset 0, enc DW_ATE_signed_char]

// CHECK: [[R]] = {{.*}}, [[CON2:![0-9]*]]} ; [ DW_TAG_reference_type ] [line 0, size 0, align 0, offset 0] [from ]
// CHECK: [[CON2]] = {{.*}}, !"_ZTS12basic_stringIcE"} ; [ DW_TAG_const_type ] [line 0, size 0, align 0, offset 0] [from _ZTS12basic_stringIcE]
// CHECK: !"0x2e\00assign\00{{.*}}\008"{{, [^,]+, [^,]+}}, !8, {{.*}} ; [ DW_TAG_subprogram ] [line 7] [def] [scope 8] [assign]
