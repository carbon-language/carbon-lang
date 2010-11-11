// RUN: %clang_cc1 -x objective-c++ -fblocks -fobjc-gc -triple x86_64-apple-darwin -O0 -S %s -o %t-64.s
// RUN: FileCheck -check-prefix LP64 --input-file=%t-64.s %s


struct S {
    int i1;
    id o1;
    struct V {
     int i2;
     id o2;
    } v1;
    int i3;
    id o3;
};

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

// Test4
    struct S s2;
    void (^e)() = ^{
        x(s2.o1);
    };    
    e();
}

// Test 5 (unions/structs and their nesting):
void Test5() {
struct S5 {
    int i1;
    id o1;
    struct V {
     int i2;
     id o2;
    } v1;
    int i3;
    union UI {
        void * i1;
        id o1;
        int i3;
        id o3;
    }ui;
};

union U {
        void * i1;
        id o1;
        int i3;
        id o3;
}ui;

struct S5 s2;
union U u2;
void (^c)() = ^{
    x(s2.ui.o1);
    x(u2.o1);
};
c();

}

// rdar: //8417746
void CFRelease(id);
void notifyBlock(id dependentBlock) {
 id singleObservationToken;
 id token;
 void (^b)();
 void (^wrapperBlock)() = ^() {
     CFRelease(singleObservationToken);
     CFRelease(singleObservationToken);
     CFRelease(token);
     CFRelease(singleObservationToken);
     b();
    };
 wrapperBlock();
}

void test_empty_block() {
 void (^wrapperBlock)() = ^() {
    };
 wrapperBlock();
}

// CHECK-LP64: L_OBJC_CLASS_NAME_:
// CHECK-LP64-NEXT: .asciz      "\0011\024"

// CHECK-LP64: L_OBJC_CLASS_NAME_1:
// CHECK-LP64-NEXT: .asciz   "\0011\025"

// CHECK-LP64: L_OBJC_CLASS_NAME_6:
// CHECK-LP64-NEXT: .asciz   "\0011\023!"

// CHECK-LP64: L_OBJC_CLASS_NAME_11:
// CHECK-LP64-NEXT: .asciz   "\001A\021\021"

// CHECK-LP64: L_OBJC_CLASS_NAME_16:
// CHECK-LP64-NEXT: .asciz   "\001A\021\022p"

// CHECK-LP64: L_OBJC_CLASS_NAME_20:
// CHECK-LP64-NEXT: .asciz   "\0013"

// CHECK-LP64: L_OBJC_CLASS_NAME_24:
// CHECK-LP64-NEXT: .asciz   "\001"
