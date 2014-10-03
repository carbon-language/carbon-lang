// RUN: %clang_cc1 -emit-llvm -g -triple x86_64-pc-linux-gnu %s -o - | FileCheck %s

struct C {
  void member_function();
  static int static_member_function();
  static int static_member_variable;
};

int C::static_member_variable = 0;

void C::member_function() { static_member_variable = 0; }

int C::static_member_function() { return static_member_variable; }

C global_variable;

int global_function() { return -1; }

namespace ns {
void global_namespace_function() { global_variable.member_function(); }
int global_namespace_variable = 1;
}

// Check that the functions that belong to C have C as a context and the
// functions that belong to the namespace have it as a context, and the global
// function has the file as a context.

// CHECK: metadata !"0x2e\00member_function\00{{.*}}", metadata !{{[0-9]+}}, metadata !"_ZTS1C"{{.*}} [ DW_TAG_subprogram ] [line 11] [def] [member_function]

// CHECK: metadata !"0x2e\00static_member_function\00{{.*}}", metadata !{{[0-9]+}}, metadata !"_ZTS1C"{{.*}}  [ DW_TAG_subprogram ] [line 13] [def] [static_member_function]

// CHECK: metadata !"0x2e\00global_function\00{{[^,]+}}", metadata !{{[0-9]+}}, metadata [[FILE:![0-9]*]]{{.*}}  [ DW_TAG_subprogram ] [line 17] [def] [global_function]
// CHECK: [[FILE]] = {{.*}} [ DW_TAG_file_type ]

// CHECK: metadata !"0x2e\00global_namespace_function\00{{[^,]+}}", metadata !{{[0-9]+}}, metadata [[NS:![0-9]*]]{{.*}} [ DW_TAG_subprogram ] [line 20] [def] [global_namespace_function]
// CHECK: [[NS]] = {{.*}} [ DW_TAG_namespace ] [ns] [line 19]
