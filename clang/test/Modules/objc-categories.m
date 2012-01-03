// RUN: rm -rf %t
// RUN: %clang_cc1 -fmodules -fmodule-cache-path %t -x objective-c -fmodule-name=category_top -emit-module %S/Inputs/module.map
// RUN: %clang_cc1 -fmodules -fmodule-cache-path %t -x objective-c -fmodule-name=category_left -emit-module %S/Inputs/module.map
// RUN: %clang_cc1 -fmodules -fmodule-cache-path %t -x objective-c -fmodule-name=category_right -emit-module %S/Inputs/module.map
// RUN: %clang_cc1 -fmodules -fmodule-cache-path %t -x objective-c -fmodule-name=category_bottom -emit-module %S/Inputs/module.map
// RUN: %clang_cc1 -fmodules -fmodule-cache-path %t %s -verify

@import category_bottom;


// in category_left.h: expected-note {{previous definition}}

@interface Foo(Source)
-(void)source; 
@end

void test(Foo *foo, LeftFoo *leftFoo) {
  [foo source];
  [foo bottom];
  [foo left];
  [foo right1];
  [foo right2];
  [foo top];

  [leftFoo left];
  [leftFoo bottom];
}
