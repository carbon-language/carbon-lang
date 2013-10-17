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

// CHECK: metadata !"_ZTS1C", metadata !"member_function"{{.*}} [ DW_TAG_subprogram ] [line 11] [def] [member_function]

// CHECK: metadata !"_ZTS1C", metadata !"static_member_function"{{.*}}  [ DW_TAG_subprogram ] [line 13] [def] [static_member_function]

// CHECK: metadata !22, metadata !"global_function"{{.*}}  [ DW_TAG_subprogram ] [line 17] [def] [global_function]
// CHECK: !22 = {{.*}} [ DW_TAG_file_type ]

// CHECK: metadata !24, metadata !"global_namespace_function"{{.*}} [ DW_TAG_subprogram ] [line 20] [def] [global_namespace_function]
// CHECK: !24 = {{.*}} [ DW_TAG_namespace ] [ns] [line 19]
