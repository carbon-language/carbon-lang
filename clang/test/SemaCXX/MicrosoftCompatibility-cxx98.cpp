// RUN: %clang_cc1 %s -triple i686-pc-win32 -fsyntax-only -std=c++98 -Wmicrosoft -verify -fms-compatibility -fexceptions -fcxx-exceptions


//MSVC allows forward enum declaration
enum ENUM; // expected-warning {{forward references to 'enum' types are a Microsoft extension}}
ENUM *var = 0;     
ENUM var2 = (ENUM)3;
enum ENUM1* var3 = 0;// expected-warning {{forward references to 'enum' types are a Microsoft extension}}

typedef void (*FnPtrTy)();
void (*PR23733_1)() = static_cast<FnPtrTy>((void *)0); // expected-warning {{static_cast between pointer-to-function and pointer-to-object is a Microsoft extension}}
void (*PR23733_2)() = FnPtrTy((void *)0);
void (*PR23733_3)() = (FnPtrTy)((void *)0);
void (*PR23733_4)() = reinterpret_cast<FnPtrTy>((void *)0);
