// RUN: rm -rf %t
// RUN: %clang_cc1 -fmodules -verify -fmodules-cache-path=%t -I %S/Inputs/template-specialization-visibility -std=c++11 %s
//
// FIXME: We should accept the explicit instantiation cases below too.
// Note, errors trigger implicit imports, so only enable one error at a time.
// RUN: %clang_cc1 -fmodules -verify -fmodules-cache-path=%t -I %S/Inputs/template-specialization-visibility -std=c++11 -DERR1 %s
// RUN: %clang_cc1 -fmodules -verify -fmodules-cache-path=%t -I %S/Inputs/template-specialization-visibility -std=c++11 -DERR2 %s
// RUN: %clang_cc1 -fmodules -verify -fmodules-cache-path=%t -I %S/Inputs/template-specialization-visibility -std=c++11 -DERR3 %s

#include "c.h"

S<int> implicit_inst_class_template;
int k1 = implicit_inst_class_template.n;

#ifdef ERR1
S<char> explicit_inst_class_template; // expected-error {{must be imported from module 'tsv.e'}}
// expected-note@e.h:4 {{previous}}
int k2 = explicit_inst_class_template.n;
#endif

#include "a.h"

T<int>::S implicit_inst_member_class_template;
int k3 = implicit_inst_member_class_template.n;

#ifdef ERR2
T<char>::S explicit_inst_member_class_template; // expected-error {{must be imported from module 'tsv.e'}}
// expected-note@e.h:5 {{previous}}
int k4 = explicit_inst_member_class_template.n;
#endif

T<int>::E implicit_inst_member_enum_template;
int k5 = decltype(implicit_inst_member_enum_template)::e;

#ifdef ERR3
T<char>::E explicit_inst_member_enum_template; // expected-error {{must be imported from module 'tsv.e'}}
// expected-note@e.h:5 {{previous}}
int k6 = decltype(explicit_inst_member_enum_template)::e;
#endif

#if ERR1 + ERR2 + ERR3 == 0
// expected-no-diagnostics
#endif
