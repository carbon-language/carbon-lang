// Test is line- and column-sensitive; see below.

struct X {
  X(int value);
  X(const X& x);
protected:
  ~X();
private:
  operator X*();
};

X::X(int value) {
}

// RUN: c-index-test -test-load-source all %s | FileCheck %s
// CHECK: load-classes.cpp:3:8: StructDecl=X:3:8 (Definition) Extent=[3:1 - 10:2]
// CHECK: load-classes.cpp:4:3: CXXConstructor=X:4:3 Extent=[4:3 - 4:15] [access=public]
// FIXME: missing TypeRef in the constructor name
// CHECK: load-classes.cpp:4:9: ParmDecl=value:4:9 (Definition) Extent=[4:5 - 4:14]
// CHECK: load-classes.cpp:5:3: CXXConstructor=X:5:3 Extent=[5:3 - 5:16] [access=public]
// FIXME: missing TypeRef in the constructor name
// CHECK: load-classes.cpp:5:14: ParmDecl=x:5:14 (Definition) Extent=[5:5 - 5:15]
// CHECK: load-classes.cpp:5:11: TypeRef=struct X:3:8 Extent=[5:11 - 5:12]
// CHECK: load-classes.cpp:7:3: CXXDestructor=~X:7:3 Extent=[7:3 - 7:7] [access=protected]
// FIXME: missing TypeRef in the destructor name
// CHECK: load-classes.cpp:9:3: CXXConversion=operator X *:9:3 Extent=[9:3 - 9:16] [access=private]
// CHECK: load-classes.cpp:9:12: TypeRef=struct X:3:8 Extent=[9:12 - 9:13]
// CHECK: load-classes.cpp:12:4: CXXConstructor=X:12:4 (Definition) Extent=[12:1 - 13:2] [access=public]
// CHECK: load-classes.cpp:12:1: TypeRef=struct X:3:8 Extent=[12:1 - 12:2]
// CHECK: load-classes.cpp:12:10: ParmDecl=value:12:10 (Definition) Extent=[12:6 - 12:15]
