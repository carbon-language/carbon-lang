// RUN: %clang_cc1 %s -triple=thumbv7-apple-darwin3.0.0-iphoneos -fno-use-cxa-atexit -target-abi apcs-gnu -emit-llvm -o - | FileCheck %s

class foo {
public:
    foo();
    virtual ~foo();
};

class bar : public foo {
public:
	bar();
};

// The global dtor needs the right calling conv with -fno-use-cxa-atexit
// rdar://7817590
bar baz;

// CHECK: @_GLOBAL__D_a()
// CHECK: call arm_apcscc  void @_ZN3barD1Ev(%class.bar* @baz)

