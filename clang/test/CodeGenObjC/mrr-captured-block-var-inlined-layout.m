// RUN: %clang_cc1 -fblocks -triple x86_64-apple-darwin -O0 -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -fblocks -triple i386-apple-darwin -O0 -emit-llvm %s -o - | FileCheck -check-prefix=CHECK-i386 %s
// rdar://12184410

void x(id y) {}
void y(int a) {}

extern id opaque_id();
__weak id wid;

void f() {
    __block int byref_int = 0;
    const id bar = (id) opaque_id();
    id baz = 0;
    __strong id strong_void_sta;
    __block id byref_bab = (id)0;
    __block id bl_var1;

// block variable layout: BL_STRONG:1, BL_OPERATOR:0
// Inline instruction for block variable layout: 0x0100
// CHECK: internal constant{{.*}}i64 256
// CHECK-i386: internal constant{{.*}}i32 256
    void (^b)() = ^{
        x(bar);
    };    

// block variable layout: BL_STRONG:2, BL_BYREF:1, BL_OPERATOR:0
// Inline instruction for block variable layout: 0x0210
// CHECK: internal constant{{.*}}i64 528
// CHECK-i386: internal constant{{.*}}i32 528
    void (^c)() = ^{
        x(bar);
        x(baz);
        byref_int = 1;
    };    

// block variable layout: BL_STRONG:2, BL_BYREF:3, BL_OPERATOR:0
// Inline instruction for block variable layout: 0x0230
// CHECK: internal constant{{.*}}i64 560
// CHECK-i386: internal constant{{.*}}i32 560
    void (^d)() = ^{
        x(bar);
        x(baz);
        byref_int = 1;
        bl_var1 = 0;
        byref_bab = 0;
    };

// block variable layout: BL_STRONG:2, BL_BYREF:3, BL_OPERATOR:0
// Inline instruction for block variable layout: 0x0230
// CHECK: internal constant{{.*}}i64 560
// CHECK-i386: internal constant{{.*}}i32 560
    id (^e)() = ^{
        x(bar);
        x(baz);
        byref_int = 1;
        bl_var1 = 0;
        byref_bab = 0;
        return wid;
    };

// Inline instruction for block variable layout: 0x020
// CHECK: internal constant{{.*}}i64 32
// CHECK-i386: internal constant{{.*}}i32 32
    void (^ii)() = ^{
       byref_int = 1;
       byref_bab = 0;
    };
}
