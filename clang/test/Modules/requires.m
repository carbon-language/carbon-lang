// RUN: rm -rf %t
// RUN: %clang_cc1 -Wauto-import -fmodules-cache-path=%t -fmodules -F %S/Inputs %s -verify -fmodule-feature custom_req1

@import DependsOnModule.CXX; // expected-error{{module 'DependsOnModule.CXX' requires feature 'cplusplus'}}
@import DependsOnModule.NotCXX;
@import DependsOnModule.NotObjC; // expected-error{{module 'DependsOnModule.NotObjC' is incompatible with feature 'objc'}}
@import DependsOnModule.CustomReq1; // OK
@import DependsOnModule.CustomReq2; // expected-error{{module 'DependsOnModule.CustomReq2' requires feature 'custom_req2'}}
