// RUN: %clang_cc1 -emit-llvm -g -triple %itanium_abi_triple %s -o - | FileCheck %s
// Test the various accessibility flags in the debug info.
struct A {
  // CHECK-DAG: [ DW_TAG_subprogram ] [line [[@LINE+1]]] [pub_default]
  void pub_default();
  // CHECK-DAG: [ DW_TAG_member ] [pub_default_static] [line [[@LINE+1]]{{.*}}offset 0] [static]
  static int pub_default_static;
};

// CHECK: [ DW_TAG_inheritance ] {{.*}} [public] [from {{.*}}A]
class B : public A {
public:
  // CHECK-DAG: [ DW_TAG_subprogram ] [line [[@LINE+1]]] [public] [pub]
  void pub();
  // CHECK-DAG: [ DW_TAG_member ] [public_static] [line [[@LINE+1]]{{.*}} [public] [static]
  static int public_static;
protected:
  // CHECK: [ DW_TAG_subprogram ] [line [[@LINE+1]]] [protected] [prot]
  void prot();
private:
  // CHECK: [ DW_TAG_subprogram ] [line [[@LINE+1]]] [priv_default]
  void priv_default();
};

union U {
  // CHECK-DAG: [ DW_TAG_subprogram ] [line [[@LINE+1]]] [union_pub_default]
  void union_pub_default();
private:
  // CHECK-DAG: [ DW_TAG_member ] [union_priv] [line [[@LINE+1]]{{.*}} [private] 
  int union_priv;
};


// CHECK: {{.*}}\00256\00{{.*}} ; [ DW_TAG_subprogram ] [line [[@LINE+1]]] [def] [free]
void free() {}

A a;
B b;
U u;
