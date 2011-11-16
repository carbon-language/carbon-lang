// RUN: rm -rf %t
// FIXME: Eventually, we should be able to remove these explicit module creation lines
// RUN: %clang_cc1 -x objective-c -fmodule-cache-path %t -fmodule-name=libA -emit-module-from-map %S/Inputs/normal-module-map/module.map
// RUN: %clang_cc1 -x objective-c -fmodule-cache-path %t -fmodule-name=libB -emit-module-from-map %S/Inputs/normal-module-map/module.map
// RUN: %clang_cc1 -x objective-c -fmodule-cache-path %t -fmodule-name=libNested -emit-module-from-map %S/Inputs/normal-module-map/nested/module.map
// RUN: %clang_cc1 -x objective-c -fmodule-cache-path %t -fauto-module-import -I %S/Inputs/normal-module-map %s -verify
#include "Umbrella/umbrella_sub.h"

int getUmbrella() { 
  return umbrella + umbrella_sub; 
}

__import_module__ Umbrella2;

#include "a1.h"
#include "b1.h"
#include "nested/nested2.h"

int test() {
  return a1 + b1 + nested2;
}
