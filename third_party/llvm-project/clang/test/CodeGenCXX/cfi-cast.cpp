// RUN: %clang_cc1 -triple x86_64-unknown-linux -fvisibility hidden -std=c++11 -fsanitize=cfi-derived-cast -fsanitize-trap=cfi-derived-cast -emit-llvm -o - %s | FileCheck -check-prefix=CHECK-DCAST %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux -fvisibility hidden -std=c++11 -fsanitize=cfi-unrelated-cast -fsanitize-trap=cfi-unrelated-cast -emit-llvm -o - %s | FileCheck -check-prefix=CHECK-UCAST %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux -fvisibility hidden -std=c++11 -fsanitize=cfi-unrelated-cast,cfi-cast-strict -fsanitize-trap=cfi-unrelated-cast,cfi-cast-strict -emit-llvm -o - %s | FileCheck -check-prefix=CHECK-UCAST-STRICT %s

// In this test the main thing we are searching for is something like
// 'metadata !"1B"' where "1B" is the mangled name of the class we are
// casting to (or maybe its base class in non-strict mode).

struct A {
  virtual void f();
  int i() const;
};

struct B : A {
  virtual void f();
};

struct C : A {};

// CHECK-DCAST-LABEL: define hidden void @_Z3abpP1A
void abp(A *a) {
  // CHECK-DCAST: [[P:%[^ ]*]] = call i1 @llvm.type.test(i8* {{%[^ ]*}}, metadata !"_ZTS1B")
  // CHECK-DCAST-NEXT: br i1 [[P]], label %[[CONTBB:[^ ]*]], label %[[TRAPBB:[^ ,]*]]

  // CHECK-DCAST: [[TRAPBB]]
  // CHECK-DCAST-NEXT: call void @llvm.ubsantrap(i8 2)
  // CHECK-DCAST-NEXT: unreachable

  // CHECK-DCAST: [[CONTBB]]
  // CHECK-DCAST: ret
  (void)static_cast<B*>(a);
}

// CHECK-DCAST-LABEL: define hidden void @_Z3abrR1A
void abr(A &a) {
  // CHECK-DCAST: [[P:%[^ ]*]] = call i1 @llvm.type.test(i8* {{%[^ ]*}}, metadata !"_ZTS1B")
  // CHECK-DCAST-NEXT: br i1 [[P]], label %[[CONTBB:[^ ]*]], label %[[TRAPBB:[^ ,]*]]

  // CHECK-DCAST: [[TRAPBB]]
  // CHECK-DCAST-NEXT: call void @llvm.ubsantrap(i8 2)
  // CHECK-DCAST-NEXT: unreachable

  // CHECK-DCAST: [[CONTBB]]
  // CHECK-DCAST: ret
  (void)static_cast<B&>(a);
}

// CHECK-DCAST-LABEL: define hidden void @_Z4abrrO1A
void abrr(A &&a) {
  // CHECK-DCAST: [[P:%[^ ]*]] = call i1 @llvm.type.test(i8* {{%[^ ]*}}, metadata !"_ZTS1B")
  // CHECK-DCAST-NEXT: br i1 [[P]], label %[[CONTBB:[^ ]*]], label %[[TRAPBB:[^ ,]*]]

  // CHECK-DCAST: [[TRAPBB]]
  // CHECK-DCAST-NEXT: call void @llvm.ubsantrap(i8 2)
  // CHECK-DCAST-NEXT: unreachable

  // CHECK-DCAST: [[CONTBB]]
  // CHECK-DCAST: ret
  (void)static_cast<B&&>(a);
}

// CHECK-UCAST-LABEL: define hidden void @_Z3vbpPv
void vbp(void *p) {
  // CHECK-UCAST: [[P:%[^ ]*]] = call i1 @llvm.type.test(i8* {{%[^ ]*}}, metadata !"_ZTS1B")
  // CHECK-UCAST-NEXT: br i1 [[P]], label %[[CONTBB:[^ ]*]], label %[[TRAPBB:[^ ,]*]]

  // CHECK-UCAST: [[TRAPBB]]
  // CHECK-UCAST-NEXT: call void @llvm.ubsantrap(i8 2)
  // CHECK-UCAST-NEXT: unreachable

  // CHECK-UCAST: [[CONTBB]]
  // CHECK-UCAST: ret
  (void)static_cast<B*>(p);
}

// CHECK-UCAST-LABEL: define hidden void @_Z3vbrRc
void vbr(char &r) {
  // CHECK-UCAST: [[P:%[^ ]*]] = call i1 @llvm.type.test(i8* {{%[^ ]*}}, metadata !"_ZTS1B")
  // CHECK-UCAST-NEXT: br i1 [[P]], label %[[CONTBB:[^ ]*]], label %[[TRAPBB:[^ ,]*]]

  // CHECK-UCAST: [[TRAPBB]]
  // CHECK-UCAST-NEXT: call void @llvm.ubsantrap(i8 2)
  // CHECK-UCAST-NEXT: unreachable

  // CHECK-UCAST: [[CONTBB]]
  // CHECK-UCAST: ret
  (void)reinterpret_cast<B&>(r);
}

// CHECK-UCAST-LABEL: define hidden void @_Z4vbrrOc
void vbrr(char &&r) {
  // CHECK-UCAST: [[P:%[^ ]*]] = call i1 @llvm.type.test(i8* {{%[^ ]*}}, metadata !"_ZTS1B")
  // CHECK-UCAST-NEXT: br i1 [[P]], label %[[CONTBB:[^ ]*]], label %[[TRAPBB:[^ ,]*]]

  // CHECK-UCAST: [[TRAPBB]]
  // CHECK-UCAST-NEXT: call void @llvm.ubsantrap(i8 2)
  // CHECK-UCAST-NEXT: unreachable

  // CHECK-UCAST: [[CONTBB]]
  // CHECK-UCAST: ret
  (void)reinterpret_cast<B&&>(r);
}

// CHECK-UCAST-LABEL: define hidden void @_Z3vcpPv
// CHECK-UCAST-STRICT-LABEL: define hidden void @_Z3vcpPv
void vcp(void *p) {
  // CHECK-UCAST: call i1 @llvm.type.test(i8* {{%[^ ]*}}, metadata !"_ZTS1A")
  // CHECK-UCAST-STRICT: call i1 @llvm.type.test(i8* {{%[^ ]*}}, metadata !"_ZTS1C")
  (void)static_cast<C*>(p);
}

// CHECK-UCAST-LABEL: define hidden void @_Z3bcpP1B
// CHECK-UCAST-STRICT-LABEL: define hidden void @_Z3bcpP1B
void bcp(B *p) {
  // CHECK-UCAST: call i1 @llvm.type.test(i8* {{%[^ ]*}}, metadata !"_ZTS1A")
  // CHECK-UCAST-STRICT: call i1 @llvm.type.test(i8* {{%[^ ]*}}, metadata !"_ZTS1C")
  (void)(C *)p;
}

// CHECK-UCAST-LABEL: define hidden void @_Z8bcp_callP1B
// CHECK-UCAST-STRICT-LABEL: define hidden void @_Z8bcp_callP1B
void bcp_call(B *p) {
  // CHECK-UCAST: call i1 @llvm.type.test(i8* {{%[^ ]*}}, metadata !"_ZTS1A")
  // CHECK-UCAST-STRICT: call i1 @llvm.type.test(i8* {{%[^ ]*}}, metadata !"_ZTS1C")
  ((C *)p)->f();
}

// CHECK-UCAST-LABEL: define hidden noundef i32 @_Z6a_callP1A
// CHECK-UCAST-STRICT-LABEL: define hidden noundef i32 @_Z6a_callP1A
int a_call(A *a) {
  // CHECK-UCAST-NOT: @llvm.type.test
  // CHECK-UCAST-STRICT-NOT: @llvm.type.test
  return a->i();
}
