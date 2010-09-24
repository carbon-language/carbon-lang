// Test that the ivars created via default synthesis are not reported.
@class NSArray, NSString, NSButton;
@interface TestA
{
  NSArray * arrayIVar;
  NSString * stringIVar;
  id idIVar;
  NSArray * arrayOutlet;
  NSString * stringOutlet;
  id idOutlet;
  NSArray *arrayOutletCollection;
}
@property(assign) NSString *stringProperty;
@property(assign) NSButton *buttonProperty;
@property(assign) NSArray *arrayOutletCollectionProperty;
+ (id) idClassMethod;
- (id) idInstanceMethod;
@end
@implementation TestA
+ (id) idClassMethod
{
    return 0;
}
- (id) idInstanceMethod
{
    return 0;
}
@end

// RUN: c-index-test -test-load-source local %s | FileCheck %s
// CHECK: properties-default-synthesis.m:2:1: UnexposedDecl=[2:8, 2:17, 2:27] Extent=[2:1 - 2:35]
// CHECK: properties-default-synthesis.m:2:8: ObjCClassRef=NSArray:2:8 Extent=[2:8 - 2:15]
// CHECK: properties-default-synthesis.m:2:17: ObjCClassRef=NSString:2:17 Extent=[2:17 - 2:25]
// CHECK: properties-default-synthesis.m:2:27: ObjCClassRef=NSButton:2:27 Extent=[2:27 - 2:35]
// CHECK: properties-default-synthesis.m:3:12: ObjCInterfaceDecl=TestA:3:12 Extent=[3:1 - 18:5]
// CHECK: properties-default-synthesis.m:5:13: ObjCIvarDecl=arrayIVar:5:13 (Definition) Extent=[5:13 - 5:22]
// CHECK: properties-default-synthesis.m:5:3: ObjCClassRef=NSArray:2:8 Extent=[5:3 - 5:10]
// CHECK: properties-default-synthesis.m:6:14: ObjCIvarDecl=stringIVar:6:14 (Definition) Extent=[6:14 - 6:24]
// CHECK: properties-default-synthesis.m:6:3: ObjCClassRef=NSString:2:17 Extent=[6:3 - 6:11]
// CHECK: properties-default-synthesis.m:7:6: ObjCIvarDecl=idIVar:7:6 (Definition) Extent=[7:6 - 7:12]
// CHECK: properties-default-synthesis.m:7:3: TypeRef=id:0:0 Extent=[7:3 - 7:5]
// CHECK: properties-default-synthesis.m:8:13: ObjCIvarDecl=arrayOutlet:8:13 (Definition) Extent=[8:13 - 8:24]
// CHECK: properties-default-synthesis.m:8:3: ObjCClassRef=NSArray:2:8 Extent=[8:3 - 8:10]
// CHECK: properties-default-synthesis.m:9:14: ObjCIvarDecl=stringOutlet:9:14 (Definition) Extent=[9:14 - 9:26]
// CHECK: properties-default-synthesis.m:9:3: ObjCClassRef=NSString:2:17 Extent=[9:3 - 9:11]
// CHECK: properties-default-synthesis.m:10:6: ObjCIvarDecl=idOutlet:10:6 (Definition) Extent=[10:6 - 10:14]
// CHECK: properties-default-synthesis.m:10:3: TypeRef=id:0:0 Extent=[10:3 - 10:5]
// CHECK: properties-default-synthesis.m:11:12: ObjCIvarDecl=arrayOutletCollection:11:12 (Definition) Extent=[11:12 - 11:33]
// CHECK: properties-default-synthesis.m:11:3: ObjCClassRef=NSArray:2:8 Extent=[11:3 - 11:10]
// CHECK: properties-default-synthesis.m:13:29: ObjCPropertyDecl=stringProperty:13:29 Extent=[13:29 - 13:43]
// CHECK: properties-default-synthesis.m:13:19: ObjCClassRef=NSString:2:17 Extent=[13:19 - 13:27]
// CHECK: properties-default-synthesis.m:14:29: ObjCPropertyDecl=buttonProperty:14:29 Extent=[14:29 - 14:43]
// CHECK: properties-default-synthesis.m:14:19: ObjCClassRef=NSButton:2:27 Extent=[14:19 - 14:27]
// CHECK: properties-default-synthesis.m:15:28: ObjCPropertyDecl=arrayOutletCollectionProperty:15:28 Extent=[15:28 - 15:57]
// CHECK: properties-default-synthesis.m:15:19: ObjCClassRef=NSArray:2:8 Extent=[15:19 - 15:26]
// CHECK: properties-default-synthesis.m:16:1: ObjCClassMethodDecl=idClassMethod:16:1 Extent=[16:1 - 16:22]
// CHECK: properties-default-synthesis.m:16:4: TypeRef=id:0:0 Extent=[16:4 - 16:6]
// CHECK: properties-default-synthesis.m:17:1: ObjCInstanceMethodDecl=idInstanceMethod:17:1 Extent=[17:1 - 17:25]
// CHECK: properties-default-synthesis.m:17:4: TypeRef=id:0:0 Extent=[17:4 - 17:6]
// CHECK: properties-default-synthesis.m:13:29: ObjCInstanceMethodDecl=stringProperty:13:29 Extent=[13:29 - 13:43]
// CHECK: properties-default-synthesis.m:13:29: ObjCInstanceMethodDecl=setStringProperty::13:29 Extent=[13:29 - 13:43]
// CHECK: properties-default-synthesis.m:13:29: ParmDecl=stringProperty:13:29 (Definition) Extent=[13:29 - 13:43]
// CHECK: properties-default-synthesis.m:14:29: ObjCInstanceMethodDecl=buttonProperty:14:29 Extent=[14:29 - 14:43]
// CHECK: properties-default-synthesis.m:14:29: ObjCInstanceMethodDecl=setButtonProperty::14:29 Extent=[14:29 - 14:43]
// CHECK: properties-default-synthesis.m:14:29: ParmDecl=buttonProperty:14:29 (Definition) Extent=[14:29 - 14:43]
// CHECK: properties-default-synthesis.m:15:28: ObjCInstanceMethodDecl=arrayOutletCollectionProperty:15:28 Extent=[15:28 - 15:57]
// CHECK: properties-default-synthesis.m:15:28: ObjCInstanceMethodDecl=setArrayOutletCollectionProperty::15:28 Extent=[15:28 - 15:57]
// CHECK: properties-default-synthesis.m:15:28: ParmDecl=arrayOutletCollectionProperty:15:28 (Definition) Extent=[15:28 - 15:57]
// CHECK: properties-default-synthesis.m:19:1: ObjCImplementationDecl=TestA:19:1 (Definition) Extent=[19:1 - 28:2]
// CHECK: properties-default-synthesis.m:20:1: ObjCClassMethodDecl=idClassMethod:20:1 (Definition) Extent=[20:1 - 23:2]
// CHECK: properties-default-synthesis.m:20:4: TypeRef=id:0:0 Extent=[20:4 - 20:6]
// CHECK: properties-default-synthesis.m:21:1: UnexposedStmt= Extent=[21:1 - 23:2]
// CHECK: properties-default-synthesis.m:22:5: UnexposedStmt= Extent=[22:5 - 22:13]
// CHECK: properties-default-synthesis.m:22:12: UnexposedExpr= Extent=[22:12 - 22:13]
// CHECK: properties-default-synthesis.m:22:12: UnexposedExpr= Extent=[22:12 - 22:13]
// CHECK: properties-default-synthesis.m:24:1: ObjCInstanceMethodDecl=idInstanceMethod:24:1 (Definition) Extent=[24:1 - 27:2]
// CHECK: properties-default-synthesis.m:24:4: TypeRef=id:0:0 Extent=[24:4 - 24:6]
// CHECK: properties-default-synthesis.m:25:1: UnexposedStmt= Extent=[25:1 - 27:2]
// CHECK: properties-default-synthesis.m:26:5: UnexposedStmt= Extent=[26:5 - 26:13]
// CHECK: properties-default-synthesis.m:26:12: UnexposedExpr= Extent=[26:12 - 26:13]
// CHECK: properties-default-synthesis.m:26:12: UnexposedExpr= Extent=[26:12 - 26:13]
// CHECK: <invalid loc>:0:0: ObjCIvarDecl=stringProperty:0:0 (Definition)
// CHECK: <invalid loc>:0:0: UnexposedDecl=:0:0 (Definition)
// CHECK: <invalid loc>:0:0: ObjCIvarDecl=arrayOutletCollectionProperty:0:0 (Definition)
// CHECK: <invalid loc>:0:0: UnexposedDecl=:0:0 (Definition)
// CHECK: <invalid loc>:0:0: ObjCIvarDecl=buttonProperty:0:0 (Definition)
// CHECK: <invalid loc>:0:0: UnexposedDecl=:0:0 (Definition)

