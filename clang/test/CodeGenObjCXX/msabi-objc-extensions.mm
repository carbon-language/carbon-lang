// RUN: %clang_cc1 -triple thumbv7-windows-msvc -fobjc-runtime=ios-6.0 -fobjc-arc -o - -emit-llvm %s | FileCheck %s

@protocol P;
@protocol Q;

@class I;
@class J<T>;

void f(id<P>, id, id<P>, id) {}
// CHECK-LABEL: "?f@@YAXPAU?$objc_object@U?$Protocol@UP@@@__ObjC@@@@PAUobjc_object@@01@Z"

void f(id, id<P>, id<P>, id) {}
// CHECK-LABEL: "?f@@YAXPAUobjc_object@@PAU?$objc_object@U?$Protocol@UP@@@__ObjC@@@@10@Z"

void f(id<P>, id<P>) {}
// CHECK-LABEL: "?f@@YAXPAU?$objc_object@U?$Protocol@UP@@@__ObjC@@@@0@Z"

void f(id<P>) {}
// CHECK-LABEL: "?f@@YAXPAU?$objc_object@U?$Protocol@UP@@@__ObjC@@@@@Z"

void f(id<P, Q>) {}
// CHECK-LABEL: "?f@@YAXPAU?$objc_object@U?$Protocol@UP@@@__ObjC@@U?$Protocol@UQ@@@2@@@@Z"

void f(Class<P>) {}
// CHECK-LABEL: "?f@@YAXPAU?$objc_class@U?$Protocol@UP@@@__ObjC@@@@@Z"

void f(Class<P, Q>) {}
// CHECK-LABEL: "?f@@YAXPAU?$objc_class@U?$Protocol@UP@@@__ObjC@@U?$Protocol@UQ@@@2@@@@Z"

void f(I<P> *) {}
// CHECK-LABEL: "?f@@YAXPAU?$I@U?$Protocol@UP@@@__ObjC@@@@@Z"

void f(I<P, Q> *) {}
// CHECK-LABEL: "?f@@YAXPAU?$I@U?$Protocol@UP@@@__ObjC@@U?$Protocol@UQ@@@2@@@@Z"

template <typename>
struct S {};

void f(S<__unsafe_unretained id>) {}
// CHECK-LABEL: "?f@@YAXU?$S@PAUobjc_object@@@@@Z"

void f(S<__autoreleasing id>) {}
// CHECK-LABEL: "?f@@YAXU?$S@U?$Autoreleasing@PAUobjc_object@@@__ObjC@@@@@Z"

void f(S<__strong id>) {}
// CHECK-LABEL: "?f@@YAXU?$S@U?$Strong@PAUobjc_object@@@__ObjC@@@@@Z"

void f(S<__weak id>) {}
// CHECK-LABEL: "?f@@YAXU?$S@U?$Weak@PAUobjc_object@@@__ObjC@@@@@Z"

void w(__weak id) {}
// CHECK-LABEL: "?w@@YAXPAUobjc_object@@@Z"

void s(__strong id) {}
// CHECK-LABEL: "?s@@YAXPAUobjc_object@@@Z"

void a(__autoreleasing id) {}
// CHECK-LABEL: "?a@@YAXPAUobjc_object@@@Z"

void u(__unsafe_unretained id) {}
// CHECK-LABEL: "?u@@YAXPAUobjc_object@@@Z"

S<__autoreleasing id> g() { return S<__autoreleasing id>(); }
// CHECK-LABEL: "?g@@YA?AU?$S@U?$Autoreleasing@PAUobjc_object@@@__ObjC@@@@XZ"

__autoreleasing id h() { return nullptr; }
// CHECK-LABEL: "?h@@YAPAUobjc_object@@XZ"

void f(I *) {}
// CHECK-LABEL: "?f@@YAXPAUI@@@Z"

void f(__kindof I *) {}
// CHECK-LABEL: "?f@@YAXPAU?$KindOf@UI@@@__ObjC@@@Z"

void f(__kindof I<P> *) {}
// CHECK-LABEL: "?f@@YAXPAU?$KindOf@U?$I@U?$Protocol@UP@@@__ObjC@@@@@__ObjC@@@Z"

void f(S<I *>) {}
// CHECK-LABEL: "?f@@YAXU?$S@U?$Strong@PAUI@@@__ObjC@@@@@Z"

void f(S<__kindof I *>) {}
// CHECK-LABEL: "?f@@YAXU?$S@U?$Strong@PAU?$KindOf@UI@@@__ObjC@@@__ObjC@@@@@Z"

void f(S<__kindof I<P> *>) {}
// CHECK-LABEL: "?f@@YAXU?$S@U?$Strong@PAU?$KindOf@U?$I@U?$Protocol@UP@@@__ObjC@@@@@__ObjC@@@__ObjC@@@@@Z"

void f(S<__weak __kindof I *>) {}
// CHECK-LABEL: "?f@@YAXU?$S@U?$Weak@PAU?$KindOf@UI@@@__ObjC@@@__ObjC@@@@@Z"

void f(S<__weak __kindof I<P> *>) {}
// CHECK-LABEL: "?f@@YAXU?$S@U?$Weak@PAU?$KindOf@U?$I@U?$Protocol@UP@@@__ObjC@@@@@__ObjC@@@__ObjC@@@@@Z"

void f(J<I *> *) {}
// CHECK-LABEL: "?f@@YAXPAU?$J@PAUI@@@@@Z"

void f(J<__kindof I *> *) {}
// CHECK-LABEL: "?f@@YAXPAU?$J@PAU?$KindOf@UI@@@__ObjC@@@@@Z"
