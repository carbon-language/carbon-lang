
// RUN: rm -rf %t
// RUN: %clang_cc1 -x objective-c -Wauto-import -fmodule-cache-path %t -fauto-module-import -F %S/Inputs %s -verify

#include <DependsOnModule/DependsOnModule.h> // expected-warning{{treating #include as an import of module 'DependsOnModule'}}

#ifdef MODULE_H_MACRO
#  error MODULE_H_MACRO should have been hidden
#endif

#ifdef DEPENDS_ON_MODULE
#  error DEPENDS_ON_MODULE should have been hidden
#endif

Module *mod; // expected-error{{unknown type name 'Module'}}

#import <AlsoDependsOnModule/AlsoDependsOnModule.h> // expected-warning{{treating #import as an import of module 'AlsoDependsOnModule'}}
Module *mod2;

int getDependsOther() { return depends_on_module_other; }
