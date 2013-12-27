// RUN: %clang_cc1 -I%S/Inputs/microsoft-header-search %s -fms-compatibility -verify

// expected-warning@Inputs/microsoft-header-search/a/findme.h:3 {{findme.h successfully included using MS search rules}}
// expected-warning@Inputs/microsoft-header-search/a/b/include3.h:3 {{#include resolved using non-portable MSVC search rules as}}

#include "Inputs/microsoft-header-search/include1.h"
