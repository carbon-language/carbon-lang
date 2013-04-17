// RUN: rm -rf %t
// RUN: %clang_cc1 -Wauto-import -fmodules-cache-path=%t -fmodules -F %S/Inputs -F %S/Inputs/DependsOnModule.framework/Frameworks %s -verify
// RUN: %clang_cc1 -x objective-c++ -Wauto-import -fmodules-cache-path=%t -fmodules -F %S/Inputs -F %S/Inputs/DependsOnModule.framework/Frameworks %s -verify

@import DependsOnModule;

void testSubFramework() {
  float *sf1 = sub_framework; // expected-error{{use of undeclared identifier 'sub_framework'}}
}

@import DependsOnModule.SubFramework;

void testSubFrameworkAgain() {
  float *sf2 = sub_framework;
  double *sfo1 = sub_framework_other;
}

#ifdef __cplusplus
@import DependsOnModule.CXX;

CXXOnly cxxonly;
#endif

@import HasSubModules;

// expected-warning@Inputs/HasSubModules.framework/Frameworks/Sub.framework/PrivateHeaders/SubPriv.h:1{{treating #include as an import of module 'HasSubModules.Sub.Types'}}
#import <HasSubModules/HasSubModulesPriv.h>

struct FrameworkSubStruct ss;
