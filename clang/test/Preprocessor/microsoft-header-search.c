// RUN: %clang_cc1 -I%S/microsoft-header-search %s -fms-compatibility -verify

// expected-warning@microsoft-header-search/a/findme.h:3 {{findme.h successfully included using MS search rules}}
// expected-warning@microsoft-header-search/a/b/include3.h:3 {{#include resolved using non-portable MSVC search rules as}}

#include "microsoft-header-search/include1.h"