// RUN: %clang_cc1 -triple x86_64-apple-darwin10.0.0 -emit-llvm -o - %s -fexceptions -std=c++11 -fblocks -fobjc-arc | FileCheck -check-prefix=ARC %s
// RUN: %clang_cc1 -triple x86_64-apple-darwin10.0.0 -emit-llvm -o - %s -fexceptions -std=c++11 -fblocks | FileCheck -check-prefix=MRC %s

typedef int (^fp)();
fp f() { auto x = []{ return 3; }; return x; }

// MRC: define internal i32 ()* @"_ZZ1fvENK3$_0cvU13block_pointerFivEEv"
// MRC: store i8* bitcast (i8** @_NSConcreteStackBlock to i8*)
// MRC: store i8* bitcast (i32 (i8*)* @"___ZZ1fvENK3$_0cvU13block_pointerFivEEv_block_invoke_0" to i8*)
// MRC: call i32 ()* (i8*, i8*)* bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to i32 ()* (i8*, i8*)*)
// MRC: call i32 ()* (i8*, i8*)* bitcast (i8* (i8*, i8*, ...)* @objc_msgSend to i32 ()* (i8*, i8*)*)
// MRC: ret i32 ()*

// ARC: define internal i32 ()* @"_ZZ1fvENK3$_0cvU13block_pointerFivEEv"
// ARC: store i8* bitcast (i8** @_NSConcreteStackBlock to i8*)
// ARC: store i8* bitcast (i32 (i8*)* @"___ZZ1fvENK3$_0cvU13block_pointerFivEEv_block_invoke_0" to i8*)
// ARC: call i8* @objc_retainBlock
// ARC: call i8* @objc_autoreleaseReturnValue
