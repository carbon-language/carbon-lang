// RUN: %clang_cc1 -fblocks -fobjc-arc -fobjc-runtime-has-weak -triple x86_64-apple-darwin -O0 -emit-llvm %s -o %t-64.s
// RUN: FileCheck -check-prefix LP64 --input-file=%t-64.s %s
// rdar://8991729

__weak id wid;
void x(id y) {}
void y(int a) {}

extern id opaque_id();

void f() {
    __block int byref_int = 0;
    char ch = 'a';
    char ch1 = 'b';
    char ch2 = 'c';
    short sh = 2;
    const id bar = (id) opaque_id();
    id baz = 0;
    __strong id strong_void_sta;
    __block id byref_bab = (id)0;
    __block id bl_var1;
    int i; double dob;

// The patterns here are a sequence of bytes, each saying first how
// many sizeof(void*) chunks to skip (high nibble) and then how many
// to scan (low nibble).  A zero byte says that we've reached the end
// of the pattern.
//
// All of these patterns start with 01 3x because the block header on
// LP64 consists of an isa pointer (which we're supposed to scan for
// some reason) followed by three words (2 ints, a function pointer,
// and a descriptor pointer).

// Test 1
// byref int, short, char, char, char, id, id, strong id, byref id
// 01 35 10 00
// CHECK-LP64: @"\01L_OBJC_CLASS_NAME_{{.*}}" = internal global [4 x i8] c"\015\10\00"
    void (^b)() = ^{
        byref_int = sh + ch+ch1+ch2 ;
        x(bar);
        x(baz);
        x((id)strong_void_sta);
        x(byref_bab);
    };    
    b();

// Test 2
// byref int, short, char, char, char, id, id, strong id, byref void*, byref id
// 01 36 10 00
// CHECK-LP64: @"\01L_OBJC_CLASS_NAME_{{.*}}" = internal global [4 x i8] c"\016\10\00"
    void (^c)() = ^{
        byref_int = sh + ch+ch1+ch2 ;
        x(bar);
        x(baz);
        x((id)strong_void_sta);
        x(wid);
        bl_var1 = 0;
        x(byref_bab);
    };    
}
