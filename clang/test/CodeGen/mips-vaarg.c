// RUN: %clang -target mips-unknown-linux     -S -o - -emit-llvm %s | FileCheck --check-prefix=ALL --check-prefix=O32 %s
// RUN: %clang -target mipsel-unknown-linux   -S -o - -emit-llvm %s | FileCheck --check-prefix=ALL --check-prefix=O32 %s
// RUN: %clang -target mips64-unknown-linux   -S -o - -emit-llvm %s -mabi=n32 | FileCheck --check-prefix=ALL --check-prefix=N32 %s
// RUN: %clang -target mips64el-unknown-linux -S -o - -emit-llvm %s -mabi=n32 | FileCheck --check-prefix=ALL --check-prefix=N32 %s
// RUN: %clang -target mips64-unknown-linux   -S -o - -emit-llvm %s -mabi=64  | FileCheck --check-prefix=ALL --check-prefix=N64 %s
// RUN: %clang -target mips64el-unknown-linux -S -o - -emit-llvm %s -mabi=64  | FileCheck --check-prefix=ALL --check-prefix=N64 %s

int foo (int a, ...)
{
    // ALL-LABEL: define i32 @foo(i32 %a, ...)

    __builtin_va_list va;
    // O32: %va = alloca i8*, align 4
    // N32: %va = alloca i8*, align 4
    // N64: %va = alloca i8*, align 8

    __builtin_va_start (va, a);
    // ALL: %va1 = bitcast i8** %va to i8*
    // ALL: call void @llvm.va_start(i8* %va1)

    int n = __builtin_va_arg (va, int);
    // ALL: %{{[0-9]+}} = va_arg i8** %va, i32
    
    __builtin_va_end (va);
    // ALL: %va2 = bitcast i8** %va to i8*
    // ALL: call void @llvm.va_end(i8* %va2)

    return n;
}
