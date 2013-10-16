// RUN: %clang_cc1 -emit-llvm %s -o - -cxx-abi microsoft \
// RUN:     -triple=i386-pc-win32 | FileCheck %s
// RUN: %clang_cc1 -emit-llvm %s -o - -triple=i386-mingw32 | FileCheck %s

void __stdcall f1(void) {}
// CHECK: define x86_stdcallcc void @"\01_f1@0"

void __fastcall f2(void) {}
// CHECK: define x86_fastcallcc void @"\01@f2@0"

void __stdcall f3() {}
// CHECK: define x86_stdcallcc void @"\01_f3@0"

void __fastcall f4(char a) {}
// CHECK: define x86_fastcallcc void @"\01@f4@4"

void __fastcall f5(short a) {}
// CHECK: define x86_fastcallcc void @"\01@f5@4"

void __fastcall f6(int a) {}
// CHECK: define x86_fastcallcc void @"\01@f6@4"

void __fastcall f7(long a) {}
// CHECK: define x86_fastcallcc void @"\01@f7@4"

void __fastcall f8(long long a) {}
// CHECK: define x86_fastcallcc void @"\01@f8@8"

void __fastcall f9(long long a, char b, char c, short d) {}
// CHECK: define x86_fastcallcc void @"\01@f9@20"(i64 %a, i8 signext %b, i8
// signext %c, i16 signext %d)

void f12(void) {}
// CHECK: define void @f12(
