// Test that @properties within class extensions are visited by
// clang_visitChildren only in the class extension, not the original
// @interface (where we have a duplicate declaration - to be removed).
@interface Foo {} @end
@interface Foo (Cat)
  @property int a;
@end
@interface Foo ()
  @property int b;
  - (void) bar;
@end

// RUN: c-index-test -test-load-source local %s | FileCheck %s
// CHECK: properties-class-extensions.m:4:12: ObjCInterfaceDecl=Foo:4:12 Extent=[4:1 - 4:23]
// CHECK: properties-class-extensions.m:5:12: ObjCCategoryDecl=Cat:5:12 Extent=[5:1 - 7:5]
// CHECK: properties-class-extensions.m:5:12: ObjCClassRef=Foo:4:12 Extent=[5:12 - 5:15]
// CHECK: properties-class-extensions.m:6:17: ObjCPropertyDecl=a:6:17 Extent=[6:17 - 6:18]
// CHECK: properties-class-extensions.m:6:17: ObjCInstanceMethodDecl=a:6:17 Extent=[6:17 - 6:18]
// CHECK: properties-class-extensions.m:6:17: ObjCInstanceMethodDecl=setA::6:17 Extent=[6:17 - 6:18]
// CHECK: properties-class-extensions.m:6:17: ParmDecl=a:6:17 (Definition) Extent=[6:17 - 6:18]
// CHECK: properties-class-extensions.m:8:12: ObjCCategoryDecl=:8:12 Extent=[8:1 - 11:5]
// CHECK: properties-class-extensions.m:8:12: ObjCClassRef=Foo:4:12 Extent=[8:12 - 8:15]
// CHECK: properties-class-extensions.m:9:17: ObjCPropertyDecl=b:9:17 Extent=[9:17 - 9:18]
// CHECK: properties-class-extensions.m:9:17: ObjCInstanceMethodDecl=b:9:17 Extent=[9:17 - 9:18]
// CHECK: properties-class-extensions.m:9:17: ObjCInstanceMethodDecl=setB::9:17 Extent=[9:17 - 9:18]
// CHECK: properties-class-extensions.m:9:17: ParmDecl=b:9:17 (Definition) Extent=[9:17 - 9:18]
// CHECK: properties-class-extensions.m:10:3: ObjCInstanceMethodDecl=bar:10:3 Extent=[10:3 - 10:16]

