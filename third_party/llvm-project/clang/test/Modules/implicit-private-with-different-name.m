// RUN: rm -rf %t

// Build PCH using A, with adjacent private module APrivate, which winds up being implicitly referenced
// RUN: %clang_cc1 -verify -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -F %S/Inputs/implicit-private-with-different-name -emit-pch -o %t-A.pch %s -Wprivate-module

// Use the PCH with no explicit way to resolve APrivate, still pick it up by automatic second-chance search for "A" with "Private" appended
// RUN: %clang_cc1 -verify -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -F %S/Inputs/implicit-private-with-different-name -include-pch %t-A.pch %s -fsyntax-only -Wprivate-module

// Check the fixit
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -F %S/Inputs/implicit-private-with-different-name -include-pch %t-A.pch %s -fsyntax-only -fdiagnostics-parseable-fixits -Wprivate-module %s 2>&1 | FileCheck %s

// expected-warning@Inputs/implicit-private-with-different-name/A.framework/Modules/module.private.modulemap:1{{expected canonical name for private module 'APrivate'}}
// expected-note@Inputs/implicit-private-with-different-name/A.framework/Modules/module.private.modulemap:1{{rename 'APrivate' to ensure it can be found by name}}
// CHECK: fix-it:"{{.*}}module.private.modulemap":{1:18-1:26}:"A_Private"

#ifndef HEADER
#define HEADER
#import "A/aprivate.h"
const int *y = &APRIVATE;
#endif
