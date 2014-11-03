// REQUIRES: powerpc-registered-target
// REQUIRES: asserts
// RUN: %clang_cc1 -triple powerpc64-unknown-linux-gnu -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple powerpc-unknown-linux-gnu -emit-llvm -o - %s | FileCheck %s --check-prefix=CHECK-PPC

#include <stdarg.h>

struct x {
  long a;
  double b;
};

void testva (int n, ...)
{
  va_list ap;

  struct x t = va_arg (ap, struct x);
// CHECK: bitcast i8* %{{[a-z.0-9]*}} to %struct.x*
// CHECK: bitcast %struct.x* %t to i8*
// CHECK: bitcast %struct.x* %{{[0-9]+}} to i8*
// CHECK: call void @llvm.memcpy
// CHECK-PPC:  %arraydecay = getelementptr inbounds [1 x %struct.__va_list_tag]* %ap, i32 0, i32 0
// CHECK-PPC-NEXT:  %gprptr = bitcast %struct.__va_list_tag* %arraydecay to i8*
// CHECK-PPC-NEXT:  %0 = ptrtoint i8* %gprptr to i32
// CHECK-PPC-NEXT:  %1 = add i32 %0, 1
// CHECK-PPC-NEXT:  %2 = inttoptr i32 %1 to i8*
// CHECK-PPC-NEXT:  %3 = add i32 %1, 3
// CHECK-PPC-NEXT:  %4 = inttoptr i32 %3 to i8**
// CHECK-PPC-NEXT:  %5 = add i32 %3, 4
// CHECK-PPC-NEXT:  %6 = inttoptr i32 %5 to i8**
// CHECK-PPC-NEXT:  %gpr = load i8* %gprptr
// CHECK-PPC-NEXT:  %fpr = load i8* %2
// CHECK-PPC-NEXT:  %overflow_area = load i8** %4
// CHECK-PPC-NEXT:  %7 = ptrtoint i8* %overflow_area to i32
// CHECK-PPC-NEXT:  %regsave_area = load i8** %6
// CHECK-PPC-NEXT:  %8 = ptrtoint i8* %regsave_area to i32
// CHECK-PPC-NEXT:  %cond = icmp ult i8 %gpr, 8
// CHECK-PPC-NEXT:  %9 = mul i8 %gpr, 4
// CHECK-PPC-NEXT:  %10 = sext i8 %9 to i32
// CHECK-PPC-NEXT:  %11 = add i32 %8, %10
// CHECK-PPC-NEXT:  br i1 %cond, label %using_regs, label %using_overflow
//
// CHECK-PPC-LABEL:using_regs:                                       ; preds = %entry
// CHECK-PPC-NEXT:  %12 = inttoptr i32 %11 to %struct.x*
// CHECK-PPC-NEXT:  %13 = add i8 %gpr, 1
// CHECK-PPC-NEXT:  store i8 %13, i8* %gprptr
// CHECK-PPC-NEXT:  br label %cont
//
// CHECK-PPC-LABEL:using_overflow:                                   ; preds = %entry
// CHECK-PPC-NEXT:  %14 = inttoptr i32 %7 to %struct.x*
// CHECK-PPC-NEXT:  %15 = add i32 %7, 4
// CHECK-PPC-NEXT:  %16 = inttoptr i32 %15 to i8*
// CHECK-PPC-NEXT:  store i8* %16, i8** %4
// CHECK-PPC-NEXT:  br label %cont
//
// CHECK-PPC-LABEL:cont:                                             ; preds = %using_overflow, %using_regs
// CHECK-PPC-NEXT:  %vaarg.addr = phi %struct.x* [ %12, %using_regs ], [ %14, %using_overflow ]
// CHECK-PPC-NEXT:  %aggrptr = bitcast %struct.x* %vaarg.addr to i8**
// CHECK-PPC-NEXT:  %aggr = load i8** %aggrptr
// CHECK-PPC-NEXT:  %17 = bitcast %struct.x* %t to i8*
// CHECK-PPC-NEXT:  call void @llvm.memcpy.p0i8.p0i8.i32(i8* %17, i8* %aggr, i32 16, i32 8, i1 false)

  int v = va_arg (ap, int);
// CHECK: ptrtoint i8* %{{[a-z.0-9]*}} to i64
// CHECK: add i64 %{{[0-9]+}}, 4
// CHECK: inttoptr i64 %{{[0-9]+}} to i8*
// CHECK: bitcast i8* %{{[0-9]+}} to i32*
// CHECK-PPC:  %arraydecay1 = getelementptr inbounds [1 x %struct.__va_list_tag]* %ap, i32 0, i32 0
// CHECK-PPC-NEXT:  %gprptr2 = bitcast %struct.__va_list_tag* %arraydecay1 to i8*
// CHECK-PPC-NEXT:  %18 = ptrtoint i8* %gprptr2 to i32
// CHECK-PPC-NEXT:  %19 = add i32 %18, 1
// CHECK-PPC-NEXT:  %20 = inttoptr i32 %19 to i8*
// CHECK-PPC-NEXT:  %21 = add i32 %19, 3
// CHECK-PPC-NEXT:  %22 = inttoptr i32 %21 to i8**
// CHECK-PPC-NEXT:  %23 = add i32 %21, 4
// CHECK-PPC-NEXT:  %24 = inttoptr i32 %23 to i8**
// CHECK-PPC-NEXT:  %gpr3 = load i8* %gprptr2
// CHECK-PPC-NEXT:  %fpr4 = load i8* %20
// CHECK-PPC-NEXT:  %overflow_area5 = load i8** %22
// CHECK-PPC-NEXT:  %25 = ptrtoint i8* %overflow_area5 to i32
// CHECK-PPC-NEXT:  %regsave_area6 = load i8** %24
// CHECK-PPC-NEXT:  %26 = ptrtoint i8* %regsave_area6 to i32
// CHECK-PPC-NEXT:  %cond7 = icmp ult i8 %gpr3, 8
// CHECK-PPC-NEXT:  %27 = mul i8 %gpr3, 4
// CHECK-PPC-NEXT:  %28 = sext i8 %27 to i32
// CHECK-PPC-NEXT:  %29 = add i32 %26, %28
// CHECK-PPC-NEXT:  br i1 %cond7, label %using_regs8, label %using_overflow9
//
// CHECK-PPC-LABEL:using_regs8:                                      ; preds = %cont
// CHECK-PPC-NEXT:  %30 = inttoptr i32 %29 to i32*
// CHECK-PPC-NEXT:  %31 = add i8 %gpr3, 1
// CHECK-PPC-NEXT:  store i8 %31, i8* %gprptr2
// CHECK-PPC-NEXT:  br label %cont10
//
// CHECK-PPC-LABEL:using_overflow9:                                  ; preds = %cont
// CHECK-PPC-NEXT:  %32 = inttoptr i32 %25 to i32*
// CHECK-PPC-NEXT:  %33 = add i32 %25, 4
// CHECK-PPC-NEXT:  %34 = inttoptr i32 %33 to i8*
// CHECK-PPC-NEXT:  store i8* %34, i8** %22
// CHECK-PPC-NEXT:  br label %cont10
//
// CHECK-PPC-LABEL:cont10:                                           ; preds = %using_overflow9, %using_regs8
// CHECK-PPC-NEXT:  %vaarg.addr11 = phi i32* [ %30, %using_regs8 ], [ %32, %using_overflow9 ]
// CHECK-PPC-NEXT:  %35 = load i32* %vaarg.addr11
// CHECK-PPC-NEXT:  store i32 %35, i32* %v, align 4

#ifdef __powerpc64__
  __int128_t u = va_arg (ap, __int128_t);
#endif
// CHECK: bitcast i8* %{{[a-z.0-9]+}} to i128*
// CHECK-NEXT: load i128* %{{[0-9]+}}
}
