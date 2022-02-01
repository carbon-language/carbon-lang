// RUN: %clang_cc1 -x c -fblocks -debug-info-kind=standalone -emit-llvm -O0 \
// RUN:   -triple x86_64-apple-darwin -o - %s | FileCheck %s
// RUN: %clang_cc1 -x c -fblocks -debug-info-kind=standalone -emit-llvm -O1 \
// RUN:   -triple x86_64-apple-darwin -o - %s \
// RUN:   | FileCheck --check-prefix=CHECK-OPT %s

// CHECK: define internal void @__f_block_invoke(i8* %.block_descriptor)
// CHECK: %.block_descriptor.addr = alloca i8*, align 8
// CHECK: %block.addr = alloca <{ i8*, i32, i32, i8*, %struct.__block_descriptor* }>*, align 8
// CHECK: store i8* %.block_descriptor, i8** %.block_descriptor.addr, align 8
// CHECK: call void @llvm.dbg.declare(metadata i8** %.block_descriptor.addr,
// CHECK-SAME:                        metadata !DIExpression())
// CHECK-OPT-NOT: alloca
// CHECK-OPT: call void @llvm.dbg.value(metadata i8* %.block_descriptor,
// CHECK-OPT-SAME:                      metadata !DIExpression())
void f() {
  a(^{
    b();
  });
}
