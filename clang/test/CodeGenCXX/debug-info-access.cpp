// RUN: %clang_cc1 -emit-llvm -g -triple %itanium_abi_triple %s -o - | FileCheck %s
// Test the various accessibility flags in the debug info.
struct A {
  // CHECK-DAG: !MDSubprogram(name: "pub_default",{{.*}} line: [[@LINE+1]],{{.*}} flags: DIFlagPrototyped,
  void pub_default();
  // CHECK-DAG: !MDDerivedType(tag: DW_TAG_member, name: "pub_default_static",{{.*}} line: [[@LINE+1]],{{.*}} flags: DIFlagStaticMember)
  static int pub_default_static;
};

// CHECK: !MDDerivedType(tag: DW_TAG_inheritance,{{.*}} baseType: !"_ZTS1A",{{.*}} flags: DIFlagPublic)
class B : public A {
public:
  // CHECK-DAG: !MDSubprogram(name: "pub",{{.*}} line: [[@LINE+1]],{{.*}} flags: DIFlagPublic | DIFlagPrototyped,
  void pub();
  // CHECK-DAG: !MDDerivedType(tag: DW_TAG_member, name: "public_static",{{.*}} line: [[@LINE+1]],{{.*}} flags: DIFlagPublic | DIFlagStaticMember)
  static int public_static;
protected:
  // CHECK: !MDSubprogram(name: "prot",{{.*}} line: [[@LINE+1]],{{.*}} flags: DIFlagProtected | DIFlagPrototyped,
  void prot();
private:
  // CHECK: !MDSubprogram(name: "priv_default",{{.*}} line: [[@LINE+1]],{{.*}} flags: DIFlagPrototyped,
  void priv_default();
};

union U {
  // CHECK-DAG: !MDSubprogram(name: "union_pub_default",{{.*}} line: [[@LINE+1]],{{.*}} flags: DIFlagPrototyped,
  void union_pub_default();
private:
  // CHECK-DAG: !MDDerivedType(tag: DW_TAG_member, name: "union_priv",{{.*}} line: [[@LINE+1]],{{.*}} flags: DIFlagPrivate)
  int union_priv;
};


// CHECK: !MDSubprogram(name: "free",
// CHECK-SAME:          isDefinition: true
// CHECK-SAME:          flags: DIFlagPrototyped,
void free() {}

A a;
B b;
U u;
