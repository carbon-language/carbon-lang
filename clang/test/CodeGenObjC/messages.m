// RUN: %clang_cc1 -emit-llvm -o - %s | FileCheck %s -check-prefix=CHECK-MAC
// RUN: %clang_cc1 -emit-llvm -fobjc-nonfragile-abi -o - %s | FileCheck %s -check-prefix=CHECK-MAC-NF
// RUN: %clang_cc1 -fgnu-runtime -emit-llvm -o - %s | FileCheck %s -check-prefix=CHECK-GNU
// RUN: %clang_cc1 -fgnu-runtime -fobjc-nonfragile-abi -emit-llvm -o - %s | FileCheck %s -check-prefix CHECK-GNU-NF

typedef struct {
  int x;
  int y;
  int z[10];
} MyPoint;

void f0(id a) {
  int i;
  MyPoint pt = { 1, 2};

  // CHECK-MAC: call {{.*}} @objc_msgSend to
  // CHECK-MAC-NF: call {{.*}} @objc_msgSend to
  // CHECK-GNU: call {{.*}} @objc_msg_lookup(
  // CHECK-GNU-NF: call {{.*}} @objc_msg_lookup_sender(
  [a print0];

  // CHECK-MAC: call {{.*}} @objc_msgSend to
  // CHECK-MAC-NF: call {{.*}} @objc_msgSend to
  // CHECK-GNU: call {{.*}} @objc_msg_lookup(
  // CHECK-GNU-NF: call {{.*}} @objc_msg_lookup_sender(
  [a print1: 10];

  // CHECK-MAC: call {{.*}} @objc_msgSend to
  // CHECK-MAC-NF: call {{.*}} @objc_msgSend to
  // CHECK-GNU: call {{.*}} @objc_msg_lookup(
  // CHECK-GNU-NF: call {{.*}} @objc_msg_lookup_sender(
  [a print2: 10 and: "hello" and: 2.2];

  // CHECK-MAC: call {{.*}} @objc_msgSend to
  // CHECK-MAC-NF: call {{.*}} @objc_msgSend to
  // CHECK-GNU: call {{.*}} @objc_msg_lookup(
  // CHECK-GNU-NF: call {{.*}} @objc_msg_lookup_sender(
  [a takeStruct: pt ];
  
  void *s = @selector(print0);
  for (i=0; i<2; ++i)
    // CHECK-MAC: call {{.*}} @objc_msgSend to
    // CHECK-MAC-NF: call {{.*}} @objc_msgSend to
    // CHECK-GNU: call {{.*}} @objc_msg_lookup(
    // CHECK-GNU-NF: call {{.*}} @objc_msg_lookup_sender(
    [a performSelector:s];
}
