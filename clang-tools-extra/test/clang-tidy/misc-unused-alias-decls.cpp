// RUN: $(dirname %s)/check_clang_tidy.sh %s misc-unused-alias-decls %t
// REQUIRES: shell

namespace my_namespace {
class C {};
}

namespace unused_alias = ::my_namespace; // eol-comments aren't removed (yet)
// CHECK-MESSAGES: :[[@LINE-1]]:11: warning: this namespace alias decl is unused
// CHECK-FIXES: {{^}}// eol-comments aren't removed (yet)

namespace used_alias = ::my_namespace;
void f() { used_alias::C c; }
