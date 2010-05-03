// RUN: c-index-test -test-load-source local %s | FileCheck %s

// From: <rdar://problem/7568881>
// The method 'bar' was also being reported outside the @implementation

@interface Foo {
  id x;
}
- (id) bar;
@end

@implementation Foo
- (id) bar {
  return 0;
}
@end

// CHECK: local-symbols.m:6:12: ObjCInterfaceDecl=Foo:6:12 Extent=[6:1 - 10:5]
// CHECK: local-symbols.m:7:6: ObjCIvarDecl=x:7:6 (Definition) Extent=[7:6 - 7:7]
// CHECK: local-symbols.m:7:3: TypeRef=id:0:0 Extent=[7:3 - 7:5]
// CHECK: local-symbols.m:9:1: ObjCInstanceMethodDecl=bar:9:1 Extent=[9:1 - 9:12]
// CHECK: local-symbols.m:9:4: TypeRef=id:0:0 Extent=[9:4 - 9:6]
// CHECK: local-symbols.m:12:1: ObjCImplementationDecl=Foo:12:1 (Definition) Extent=[12:1 - 16:2]
// CHECK: local-symbols.m:13:1: ObjCInstanceMethodDecl=bar:13:1 (Definition) Extent=[13:1 - 15:2]
// CHECK: local-symbols.m:13:4: TypeRef=id:0:0 Extent=[13:4 - 13:6]

