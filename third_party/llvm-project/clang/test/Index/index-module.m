
#include <DependsOnModule/DependsOnModule.h>
@import DependsOnModule;
int glob;

// RUN: rm -rf %t.cache %t.cache.sys
// RUN: c-index-test -index-file %s -fmodules-cache-path=%t.cache -fmodules -F %S/../Modules/Inputs \
// RUN:      -Xclang -fdisable-module-hash | FileCheck %s
// RUN: c-index-test -index-file %s -fmodules-cache-path=%t.cache.sys -fmodules -iframework %S/../Modules/Inputs \
// RUN:      -Xclang -fdisable-module-hash | FileCheck %s
// RUN: c-index-test -index-file %s -fmodules-cache-path=%t.cache -fmodules -gmodules -F %S/../Modules/Inputs \
// RUN:      -Xclang -fdisable-module-hash | FileCheck %s

// CHECK-NOT: [indexDeclaration]
// CHECK: [ppIncludedFile]: {{.*}}/Modules/Inputs/DependsOnModule.framework{{[/\\]}}Headers{{[/\\]}}DependsOnModule.h | name: "DependsOnModule/DependsOnModule.h" | hash loc: 2:1 | isImport: 0 | isAngled: 1 | isModule: 1
// CHECK-NEXT: [importedASTFile]: [[PCM:.*[/\\]DependsOnModule\.pcm]] | loc: 2:1 | name: "DependsOnModule" | isImplicit: 1
// CHECK-NOT: [indexDeclaration]
// CHECK: [importedASTFile]: [[PCM]] | loc: 3:1 | name: "DependsOnModule" | isImplicit: 0
// CHECK-NEXT: [indexDeclaration]: kind: variable | name: glob | {{.*}} | loc: 4:5 
// CHECK-NOT: [indexDeclaration]

// RUN: c-index-test -index-tu %t.cache/DependsOnModule.pcm | FileCheck %s -check-prefix=CHECK-DMOD
// RUN: c-index-test -index-tu %t.cache.sys/DependsOnModule.pcm | FileCheck %s -check-prefix=CHECK-DMOD-AST

// CHECK-DMOD:      [startedTranslationUnit]
// CHECK-DMOD-NEXT: [ppIncludedFile]: [[DMOD_MODULE_H:.*/Modules/Inputs/DependsOnModule\.framework[/\\]Headers[/\\]DependsOnModule\.h]] | {{.*}} | hash loc: <invalid> | {{.*}} | module: DependsOnModule
// CHECK-DMOD-NEXT: [ppIncludedFile]: {{.*}}/Modules/Inputs/Module.framework{{[/\\]}}Headers{{[/\\]}}Module.h | name: "Module/Module.h" | hash loc: {{.*}}/Modules/Inputs/DependsOnModule.framework{{[/\\]}}Headers{{[/\\]}}DependsOnModule.h:1:1 | isImport: 0 | isAngled: 1 | isModule: 1 | module: Module
// CHECK-DMOD-NEXT: [ppIncludedFile]: [[DMOD_OTHER_H:.*/Modules/Inputs/DependsOnModule\.framework[/\\]Headers[/\\]other\.h]] | {{.*}} | hash loc: <invalid> | {{.*}} | module: DependsOnModule
// CHECK-DMOD-NEXT: [ppIncludedFile]: [[DMOD_NOT_CXX_H:.*/Modules/Inputs/DependsOnModule\.framework[/\\]Headers[/\\]not_cxx\.h]] | {{.*}} | hash loc: <invalid> | {{.*}} | module: DependsOnModule.NotCXX
// CHECK-DMOD-NEXT: [ppIncludedFile]: [[DMOD_NOT_CORO_H:.*/Modules/Inputs/DependsOnModule\.framework[/\\]Headers[/\\]not_coroutines\.h]] | {{.*}} | hash loc: <invalid> | {{.*}} | module: DependsOnModule.NotCoroutines
// CHECK-DMOD-NEXT: [ppIncludedFile]: [[DMOD_SUB_H:.*/Modules/Inputs/DependsOnModule\.framework[/\\]Frameworks[/\\]SubFramework\.framework[/\\]Headers[/\\]SubFramework\.h]] | {{.*}} | hash loc: <invalid> | {{.*}} | module: DependsOnModule.SubFramework
// CHECK-DMOD-NEXT: [ppIncludedFile]: [[DMOD_SUB_OTHER_H:.*/Modules/Inputs/DependsOnModule.framework[/\\]Frameworks[/\\]SubFramework\.framework[/\\]Headers[/\\]Other\.h]] | name: "SubFramework/Other.h" | hash loc: [[DMOD_SUB_H]]:1:1 | isImport: 0 | isAngled: 0 | isModule: 0 | module: DependsOnModule.SubFramework.Other
// CHECK-DMOD-NEXT: [ppIncludedFile]: [[DMOD_PRIVATE_H:.*/Modules/Inputs/DependsOnModule.framework[/\\]PrivateHeaders[/\\]DependsOnModulePrivate.h]] | {{.*}} | hash loc: <invalid> | {{.*}} | module: DependsOnModule.Private.DependsOnModule
// CHECK-DMOD-NEXT: [importedASTFile]: {{.*}}.cache{{(.sys)?[/\\]}}Module.pcm | loc: [[DMOD_MODULE_H]]:1:1 | name: "Module" | isImplicit: 1
// CHECK-DMOD-NEXT: [indexDeclaration]: kind: variable | name: depends_on_module_other | {{.*}} | loc: [[DMOD_OTHER_H]]:1:5
// CHECK-DMOD-NEXT: [indexDeclaration]: kind: variable | name: template | {{.*}} | loc: [[DMOD_NOT_CXX_H]]:1:12
// CHECK-DMOD-NEXT: [indexDeclaration]: kind: variable | name: sub_framework | {{.*}} | loc: [[DMOD_SUB_H]]:2:8
// CHECK-DMOD-NEXT: [indexDeclaration]: kind: variable | name: sub_framework_other | {{.*}} | loc: [[DMOD_SUB_OTHER_H]]:1:9
// CHECK-DMOD-NEXT: [indexDeclaration]: kind: variable | name: depends_on_module_private | {{.*}} | loc: [[DMOD_PRIVATE_H]]:1:5
// CHECK-DMOD-NOT: [indexDeclaration]

// CHECK-DMOD-AST: [importedASTFile]: {{.*}}.cache.sys{{[/\\]}}Module.pcm | loc: {{.*}}DependsOnModule.h:1:1 | name: "Module" | isImplicit: 1

// RUN: c-index-test -index-tu %t.cache/Module.pcm | FileCheck %s -check-prefix=CHECK-TMOD

// CHECK-TMOD:      [startedTranslationUnit]
// CHECK-TMOD-NEXT: [ppIncludedFile]: [[TMOD_MODULE_H:.*/Modules/Inputs/Module\.framework[/\\]Headers[/\\]Module\.h]] | {{.*}} | hash loc: <invalid>
// CHECK-TMOD-NEXT: [ppIncludedFile]: [[TMODHDR:.*/Modules/Inputs/Module.framework[/\\]Headers.]]Sub.h | name: "Module/Sub.h" | hash loc: [[TMOD_MODULE_H]]:23:1 | isImport: 0 | isAngled: 1
// CHECK-TMOD-NEXT: [ppIncludedFile]: [[TMODHDR]]Sub2.h | name: "Module/Sub2.h" | hash loc: [[TMODHDR]]Sub.h:1:1 | isImport: 0 | isAngled: 1
// CHECK-TMOD-NEXT: [ppIncludedFile]: [[TMODHDR]]Buried{{[/\\]}}Treasure.h | name: "Module/Buried/Treasure.h" | hash loc: [[TMOD_MODULE_H]]:24:1 | isImport: 0 | isAngled: 1
// CHECK-TMOD-NEXT: [ppIncludedFile]: [[TMOD_SUB_H:.*[/\\]Modules[/\\]Inputs[/\\]Module\.framework[/\\]Frameworks[/\\]SubFramework\.framework[/\\]Headers[/\\]SubFramework\.h]] | {{.*}} | hash loc: <invalid>
// CHECK-TMOD-NEXT: [indexDeclaration]: kind: function | name: getModuleVersion | {{.*}} | loc: [[TMOD_MODULE_H]]:9:13
// CHECK-TMOD-NEXT: [indexDeclaration]: kind: objc-class | name: Module | {{.*}} | loc: [[TMOD_MODULE_H]]:15:12
// CHECK-TMOD-NEXT:      <ObjCContainerInfo>: kind: interface
// CHECK-TMOD-NEXT: [indexDeclaration]: kind: objc-class-method | name: version | {{.*}} | loc: [[TMOD_MODULE_H]]:16:1
// CHECK-TMOD-NEXT: [indexDeclaration]: kind: objc-class-method | name: alloc | {{.*}} | loc: [[TMOD_MODULE_H]]:17:2
// CHECK-TMOD-NEXT: [indexDeclaration]: kind: typedef | name: FILE | {{.*}} | loc: [[TMOD_MODULE_H]]:30:3
// CHECK-TMOD-NEXT: [indexDeclaration]: kind: struct | name: __sFILE | {{.*}} | loc: [[TMOD_MODULE_H]]:28:16
// CHECK-TMOD-NEXT: [indexDeclaration]: kind: field | name: _offset | {{.*}} | loc: [[TMOD_MODULE_H]]:29:7
// CHECK-TMOD-NEXT: [indexDeclaration]: kind: variable | name: myFile | {{.*}} | loc: [[TMOD_MODULE_H]]:32:14
// CHECK-TMOD-NEXT: [indexEntityReference]: kind: typedef | name: FILE | {{.*}} | loc: [[TMOD_MODULE_H]]:32:8
// CHECK-TMOD-NEXT: [indexDeclaration]: kind: variable | name: Module_Sub | {{.*}} | loc: [[TMODHDR]]Sub.h:2:6
// CHECK-TMOD-NEXT: [indexDeclaration]: kind: variable | name: Module_Sub2 | USR: c:@Module_Sub2 | {{.*}} | loc: [[TMODHDR]]Sub2.h:1:6
// CHECK-TMOD-NEXT: [indexDeclaration]: kind: variable | name: Buried_Treasure | {{.*}} | loc: [[TMODHDR]]Buried{{[/\\]}}Treasure.h:1:11
// CHECK-TMOD-NEXT: [indexDeclaration]: kind: variable | name: module_subframework | {{.*}} | loc: [[TMOD_SUB_H]]:4:7
// CHECK-TMOD-NOT: [indexDeclaration]
