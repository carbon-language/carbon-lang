__import_module__ redeclarations_left;
__import_module__ redeclarations_right;

@interface MyObject : NSObject
@end

// RUN: rm -rf %t
// RUN: %clang_cc1 -x objective-c -fmodule-cache-path %t -emit-module-from-map -fmodule-name=redeclarations_left %S/Inputs/module.map
// RUN: %clang_cc1 -x objective-c -fmodule-cache-path %t -emit-module-from-map -fmodule-name=redeclarations_right %S/Inputs/module.map
// RUN: %clang_cc1 -fmodule-cache-path %t %s -verify

