// RUN: %clang_cc1 -fblocks -fobjc-arc -fobjc-runtime-has-weak -triple x86_64-apple-darwin -print-ivar-layout -emit-llvm -o /dev/null %s > %t-64.layout
// RUN: FileCheck --input-file=%t-64.layout %s
// RUN: %clang_cc1 -fblocks -fobjc-arc -fobjc-runtime-has-weak -triple i386-apple-darwin -print-ivar-layout -emit-llvm -o /dev/null  %s > %t-32.layout
// RUN: FileCheck -check-prefix=CHECK-i386 --input-file=%t-32.layout %s
// rdar://12184410

void x(id y) {}
void y(int a) {}

extern id opaque_id();

void f() {
    __block int byref_int = 0;
    const id bar = (id) opaque_id();
    id baz = 0;
    __strong id strong_void_sta;
    __block id byref_bab = (id)0;
    __block id bl_var1;

// CHECK: Inline instruction for block variable layout: 0x0100
// CHECK-i386: Inline instruction for block variable layout: 0x0100
    void (^b)() = ^{
        x(bar);
    };    

// CHECK: Inline instruction for block variable layout: 0x0210
// CHECK-i386: Inline instruction for block variable layout: 0x0210
    void (^c)() = ^{
        x(bar);
        x(baz);
        byref_int = 1;
    };    

// CHECK: Inline instruction for block variable layout: 0x0230
// CHECK-i386: Inline instruction for block variable layout: 0x0230
    void (^d)() = ^{
        x(bar);
        x(baz);
        byref_int = 1;
        bl_var1 = 0;
        byref_bab = 0;
    };

// CHECK: Inline instruction for block variable layout: 0x0231
// CHECK-i386: Inline instruction for block variable layout: 0x0231
    __weak id wid;
    id (^e)() = ^{
        x(bar);
        x(baz);
        byref_int = 1;
        bl_var1 = 0;
        byref_bab = 0;
        return wid;
    };

// CHECK: Inline instruction for block variable layout: 0x0235
// CHECK-i386: Inline instruction for block variable layout: 0x0235
    __weak id wid1, wid2, wid3, wid4;
    id (^f)() = ^{
        x(bar);
        x(baz);
        byref_int = 1;
        bl_var1 = 0;
        byref_bab = 0;
        x(wid1);
        x(wid2);
        x(wid3);
        x(wid4);
        return wid;
    };

// CHECK: Inline instruction for block variable layout: 0x035
// CHECK-i386: Inline instruction for block variable layout: 0x035
    id (^g)() = ^{
        byref_int = 1;
        bl_var1 = 0;
        byref_bab = 0;
        x(wid1);
        x(wid2);
        x(wid3);
        x(wid4);
        return wid;
    };

// CHECK: Inline instruction for block variable layout: 0x01
// CHECK-i386: Inline instruction for block variable layout: 0x01
    id (^h)() = ^{
        return wid;
    };

// CHECK: Inline instruction for block variable layout: 0x020
// CHECK-i386: Inline instruction for block variable layout: 0x020
    void (^ii)() = ^{
       byref_int = 1;
       byref_bab = 0;
    };

// CHECK: Inline instruction for block variable layout: 0x0102
// CHECK-i386: Inline instruction for block variable layout: 0x0102
    void (^jj)() = ^{
      x(bar);
      x(wid1);
      x(wid2);
    };
}

// rdar://12752901
@class NSString;
extern void NSLog(NSString *format, ...);
typedef void (^dispatch_block_t)(void);
int main() {
        __strong NSString *s1 = 0;
        __strong NSString *s2 = 0;
        __weak NSString *w1 = 0;


// CHECK: Inline instruction for block variable layout: 0x0201
// CHECK-i386: Inline instruction for block variable layout: 0x0201
        dispatch_block_t block2 = ^{
                NSLog(@"%@, %@, %@", s1, w1, s2);
        };
}
