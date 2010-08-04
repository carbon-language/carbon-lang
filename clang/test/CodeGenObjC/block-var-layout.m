// RUN: %clang_cc1 -fblocks -fobjc-gc -triple x86_64-apple-darwin -O0 -S %s -o %t-64.s
// RUN: FileCheck -check-prefix LP64 --input-file=%t-64.s %s
// RUN: %clang_cc1 -x objective-c++ -fblocks -fobjc-gc -triple x86_64-apple-darwin -O0 -S %s -o %t-64.s
// RUN: FileCheck -check-prefix LP64 --input-file=%t-64.s %s

__weak id wid;
void x(id y) {}
void y(int a) {}

void f() {
    __block int byref_int = 0;
    char ch = 'a';
    char ch1 = 'b';
    char ch2 = 'c';
    short sh = 2;
    const id bar = (id)0;
    id baz = 0;
    __strong void *strong_void_sta;
    __block id byref_bab = (id)0;
    __block void *bl_var1;
    int i; double dob;

    void (^b)() = ^{
        byref_int = sh + ch+ch1+ch2 ;
        x(bar);
        x(baz);
        x((id)strong_void_sta);
        x(byref_bab);
    };    
    b();

// Test 2
    void (^c)() = ^{
        byref_int = sh + ch+ch1+ch2 ;
        x(bar);
        x(baz);
        x((id)strong_void_sta);
        x(wid);
        bl_var1 = 0;
        x(byref_bab);
    };    
    c();

// Test 3
void (^d)() = ^{
        byref_int = sh + ch+ch1+ch2 ;
        x(bar);
        x(baz);
        x(wid);
        bl_var1 = 0; 
        y(i + dob);
        x(byref_bab);
    };    
    d();
}

// CHECK-LP64: L_OBJC_CLASS_NAME_:
// CHECK-LP64-NEXT: .asciz      "A\024"

// CHECK-LP64: L_OBJC_CLASS_NAME_1:
// CHECK-LP64-NEXT: .asciz   "A\025"

// CHECK-LP64: L_OBJC_CLASS_NAME_6:
// CHECK-LP64-NEXT: .asciz   "A\023!"

