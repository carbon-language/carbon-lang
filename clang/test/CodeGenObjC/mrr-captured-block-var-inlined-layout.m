// RUN: %clang_cc1 -fblocks -triple x86_64-apple-darwin -O0 -emit-llvm %s -o %t-64.s
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

//  Inline instruction for block variable layout: 0x0100
// CKECK-LP64: i8* getelementptr inbounds ([6 x i8]* @.str, i32 0, i32 0), i64 256 }
    void (^b)() = ^{
        x(bar);
    };    

// Inline instruction for block variable layout: 0x0210
// CKECK-LP64: i8* getelementptr inbounds ([6 x i8]* @.str, i32 0, i32 0), i64 528 }
    void (^c)() = ^{
        x(bar);
        x(baz);
        byref_int = 1;
    };    

// Inline instruction for block variable layout: 0x0230
// CKECK-LP64: i8* getelementptr inbounds ([6 x i8]* @.str, i32 0, i32 0), i64 560 }
    void (^d)() = ^{
        x(bar);
        x(baz);
        byref_int = 1;
        bl_var1 = 0;
        byref_bab = 0;
    };

// Inline instruction for block variable layout: 0x0230
// CKECK-LP64: i8* getelementptr inbounds ([6 x i8]* @.str, i32 0, i32 0), i64 560 }
    id (^e)() = ^{
        x(bar);
        x(baz);
        byref_int = 1;
        bl_var1 = 0;
        byref_bab = 0;
        return wid;
    };

// Inline instruction for block variable layout: 0x020
// CKECK-LP64: i8* getelementptr inbounds ([6 x i8]* @.str, i32 0, i32 0), i64 32 }
    void (^ii)() = ^{
       byref_int = 1;
       byref_bab = 0;
    };
}
