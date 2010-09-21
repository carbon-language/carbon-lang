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

// Test that 'setter' methods defined by @property in the class extension
// but not the in @interface are only presented in the class extension.
@interface Bar  
@property (readonly) id bar;
@end
@interface Bar () 
@property (readwrite) id bar;
@end

// RUN: c-index-test -test-load-source local %s | FileCheck %s
// CHECK: properties-class-extensions.m:4:12: ObjCInterfaceDecl=Foo:4:12 Extent=[4:1 - 4:23]
// CHECK: properties-class-extensions.m:9:15: ObjCInstanceMethodDecl=setB::9:15 Extent=[9:15 - 9:16]
// CHECK: properties-class-extensions.m:9:15: ParmDecl=b:9:15 (Definition) Extent=[9:15 - 9:16]
// CHECK: properties-class-extensions.m:5:12: ObjCCategoryDecl=Cat:5:12 Extent=[5:1 - 7:5]
// CHECK: properties-class-extensions.m:5:12: ObjCClassRef=Foo:4:12 Extent=[5:12 - 5:15]
// CHECK: properties-class-extensions.m:6:15: ObjCPropertyDecl=a:6:15 Extent=[6:15 - 6:16]
// CHECK: properties-class-extensions.m:6:15: ObjCInstanceMethodDecl=a:6:15 Extent=[6:15 - 6:16]
// CHECK: properties-class-extensions.m:6:15: ObjCInstanceMethodDecl=setA::6:15 Extent=[6:15 - 6:16]
// CHECK: properties-class-extensions.m:6:15: ParmDecl=a:6:15 (Definition) Extent=[6:15 - 6:16]
// CHECK: properties-class-extensions.m:8:12: ObjCCategoryDecl=:8:12 Extent=[8:1 - 11:5]
// CHECK: properties-class-extensions.m:8:12: ObjCClassRef=Foo:4:12 Extent=[8:12 - 8:15]
// CHECK: properties-class-extensions.m:9:15: ObjCPropertyDecl=b:9:15 Extent=[9:15 - 9:16]
// CHECK: properties-class-extensions.m:9:15: ObjCInstanceMethodDecl=b:9:15 Extent=[9:15 - 9:16]
// CHECK: properties-class-extensions.m:9:15: ObjCInstanceMethodDecl=setB::9:15 Extent=[9:15 - 9:16]
// CHECK: properties-class-extensions.m:9:15: ParmDecl=b:9:15 (Definition) Extent=[9:15 - 9:16]
// CHECK: properties-class-extensions.m:10:1: ObjCInstanceMethodDecl=bar:10:1 Extent=[10:1 - 10:14]
// CHECK: properties-class-extensions.m:15:12: ObjCInterfaceDecl=Bar:15:12 Extent=[15:1 - 17:5]
// CHECK: properties-class-extensions.m:16:25: ObjCPropertyDecl=bar:16:25 Extent=[16:25 - 16:28]
// CHECK: properties-class-extensions.m:16:22: TypeRef=id:0:0 Extent=[16:22 - 16:24]
// CHECK: properties-class-extensions.m:16:25: ObjCInstanceMethodDecl=bar:16:25 Extent=[16:25 - 16:28]
// CHECK: properties-class-extensions.m:18:12: ObjCCategoryDecl=:18:12 Extent=[18:1 - 20:5]
// CHECK: properties-class-extensions.m:18:12: ObjCClassRef=Bar:15:12 Extent=[18:12 - 18:15]
// CHECK: properties-class-extensions.m:19:26: ObjCPropertyDecl=bar:19:26 Extent=[19:26 - 19:29]
// CHECK: properties-class-extensions.m:19:23: TypeRef=id:0:0 Extent=[19:23 - 19:25]
// CHECK: properties-class-extensions.m:16:25: ObjCInstanceMethodDecl=bar:16:25 Extent=[16:25 - 16:28]
// CHECK: properties-class-extensions.m:19:26: ObjCInstanceMethodDecl=setBar::19:26 Extent=[19:26 - 19:29]
// CHECK: properties-class-extensions.m:19:26: ParmDecl=bar:19:26 (Definition) Extent=[19:26 - 19:29]


