// RUN: %clang_cc1 %s -triple=x86_64-apple-darwin10 -emit-llvm -o - | FileCheck %s
// rdar://9208606

struct MyStruct
{
	int x;
	int y;
	int z;
};

@interface MyClass
{
	MyStruct _foo;
}

@property (assign, readwrite) const MyStruct& foo;

- (const MyStruct&) foo;
- (void) setFoo:(const MyStruct&)inFoo;
@end

int main()
{
	MyClass* myClass;
	MyStruct myStruct;

	myClass.foo = myStruct;

	const MyStruct& currentMyStruct = myClass.foo;   
	return 0;
}

// CHECK: [[C:%.*]] = call %struct.MyStruct* bitcast (i8* (i8*, i8*, ...)* @objc_msgSend
// CHECK:   store %struct.MyStruct* [[C]], %struct.MyStruct** [[D:%.*]]
