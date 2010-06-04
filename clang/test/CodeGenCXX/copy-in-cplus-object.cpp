// RUN: %clang_cc1 %s -fblocks -triple x86_64-apple-darwin -emit-llvm -o - | FileCheck %s

struct TestObject
{
	TestObject(const TestObject& inObj);
	TestObject();
	TestObject& operator=(const TestObject& inObj);
	int version() const;

};

void testRoutine() {
    TestObject one;
    int (^V)() = ^{ return one.version(); };
}

// CHECK: call void @_ZN10TestObjectC1ERKS_

