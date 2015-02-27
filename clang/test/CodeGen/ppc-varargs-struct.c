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
// CHECK-PPC:  [[ARRAYDECAY:%[a-z0-9]+]] = getelementptr inbounds [1 x %struct.__va_list_tag], [1 x %struct.__va_list_tag]* %ap, i32 0, i32 0
// CHECK-PPC-NEXT:  [[GPRPTR:%[a-z0-9]+]] = bitcast %struct.__va_list_tag* [[ARRAYDECAY]] to i8*
// CHECK-PPC-NEXT:  [[ZERO:%[0-9]+]] = ptrtoint i8* [[GPRPTR]] to i32
// CHECK-PPC-NEXT:  [[ONE:%[0-9]+]] = add i32 [[ZERO]], 1
// CHECK-PPC-NEXT:  [[TWO:%[0-9]+]] = inttoptr i32 [[ONE]] to i8*
// CHECK-PPC-NEXT:  [[THREE:%[0-9]+]] = add i32 [[ONE]], 3
// CHECK-PPC-NEXT:  [[FOUR:%[0-9]+]] = inttoptr i32 [[THREE]] to i8**
// CHECK-PPC-NEXT:  [[FIVE:%[0-9]+]] = add i32 [[THREE]], 4
// CHECK-PPC-NEXT:  [[SIX:%[0-9]+]] = inttoptr i32 [[FIVE]] to i8**
// CHECK-PPC-NEXT:  [[GPR:%[a-z0-9]+]] = load i8, i8* [[GPRPTR]]
// CHECK-PPC-NEXT:  [[FPR:%[a-z0-9]+]] = load i8, i8* [[TWO]] 
// CHECK-PPC-NEXT:  [[OVERFLOW_AREA:%[a-z_0-9]+]] = load i8*, i8** [[FOUR]]
// CHECK-PPC-NEXT:  [[SEVEN:%[0-9]+]] = ptrtoint i8* [[OVERFLOW_AREA]] to i32
// CHECK-PPC-NEXT:  [[REGSAVE_AREA:%[a-z_0-9]+]] = load i8*, i8** [[SIX]]
// CHECK-PPC-NEXT:  [[EIGHT:%[0-9]+]] = ptrtoint i8* [[REGSAVE_AREA]] to i32
// CHECK-PPC-NEXT:  [[COND:%[a-z0-9]+]] = icmp ult i8 [[GPR]], 8
// CHECK-PPC-NEXT:  [[NINE:%[0-9]+]] = mul i8 [[GPR]], 4
// CHECK-PPC-NEXT:  [[TEN:%[0-9]+]] = sext i8 [[NINE]] to i32
// CHECK-PPC-NEXT:  [[ELEVEN:%[0-9]+]] = add i32 [[EIGHT]], [[TEN]]
// CHECK-PPC-NEXT:  br i1 [[COND]], label [[USING_REGS:%[a-z_0-9]+]], label [[USING_OVERFLOW:%[a-z_0-9]+]]
//
// CHECK-PPC1:[[USING_REGS]]
// CHECK-PPC:  [[TWELVE:%[0-9]+]] = inttoptr i32 [[ELEVEN]] to %struct.x*
// CHECK-PPC-NEXT:  [[THIRTEEN:%[0-9]+]] = add i8 [[GPR]], 1
// CHECK-PPC-NEXT:  store i8 [[THIRTEEN]], i8* [[GPRPTR]]
// CHECK-PPC-NEXT:  br label [[CONT:%[a-z0-9]+]]
//
// CHECK-PPC1:[[USING_OVERFLOW]]
// CHECK-PPC:  [[FOURTEEN:%[0-9]+]] = inttoptr i32 [[SEVEN]] to %struct.x*
// CHECK-PPC-NEXT:  [[FIFTEEN:%[0-9]+]] = add i32 [[SEVEN]], 4
// CHECK-PPC-NEXT:  [[SIXTEEN:%[0-9]+]] = inttoptr i32 [[FIFTEEN]] to i8*
// CHECK-PPC-NEXT:  store i8* [[SIXTEEN]], i8** [[FOUR]]
// CHECK-PPC-NEXT:  br label [[CONT]]
//
// CHECK-PPC1:[[CONT]]
// CHECK-PPC:  [[VAARG_ADDR:%[a-z.0-9]+]] = phi %struct.x* [ [[TWELVE]], [[USING_REGS]] ], [ [[FOURTEEN]], [[USING_OVERFLOW]] ]
// CHECK-PPC-NEXT:  [[AGGRPTR:%[a-z0-9]+]] = bitcast %struct.x* [[VAARG_ADDR]] to i8**
// CHECK-PPC-NEXT:  [[AGGR:%[a-z0-9]+]] = load i8*, i8** [[AGGRPTR]]
// CHECK-PPC-NEXT:  [[SEVENTEEN:%[0-9]+]] = bitcast %struct.x* %t to i8*
// CHECK-PPC-NEXT:  call void @llvm.memcpy.p0i8.p0i8.i32(i8* [[SEVENTEEN]], i8* [[AGGR]], i32 16, i32 8, i1 false)

  int v = va_arg (ap, int);
// CHECK: ptrtoint i8* %{{[a-z.0-9]*}} to i64
// CHECK: add i64 %{{[0-9]+}}, 4
// CHECK: inttoptr i64 %{{[0-9]+}} to i8*
// CHECK: bitcast i8* %{{[0-9]+}} to i32*
// CHECK-PPC:  [[ARRAYDECAY1:%[a-z0-9]+]] = getelementptr inbounds [1 x %struct.__va_list_tag], [1 x %struct.__va_list_tag]* %ap, i32 0, i32 0
// CHECK-PPC-NEXT:  [[GPRPTR1:%[a-z0-9]+]] = bitcast %struct.__va_list_tag* [[ARRAYDECAY1]] to i8*
// CHECK-PPC-NEXT:  [[EIGHTEEN:%[0-9]+]] = ptrtoint i8* [[GPRPTR1]] to i32
// CHECK-PPC-NEXT:  [[NINETEEN:%[0-9]+]] = add i32 [[EIGHTEEN]], 1
// CHECK-PPC-NEXT:  [[TWENTY:%[0-9]+]] = inttoptr i32 [[NINETEEN]] to i8*
// CHECK-PPC-NEXT:  [[TWENTYONE:%[0-9]+]] = add i32 [[NINETEEN]], 3
// CHECK-PPC-NEXT:  [[TWENTYTWO:%[0-9]+]] = inttoptr i32 [[TWENTYONE]] to i8**
// CHECK-PPC-NEXT:  [[TWENTYTHREE:%[0-9]+]] = add i32 [[TWENTYONE]], 4
// CHECK-PPC-NEXT:  [[TWENTYFOUR:%[0-9]+]] = inttoptr i32 [[TWENTYTHREE]] to i8**
// CHECK-PPC-NEXT:  [[GPR1:%[a-z0-9]+]] = load i8, i8* [[GPRPTR1]]
// CHECK-PPC-NEXT:  [[FPR1:%[a-z0-9]+]] = load i8, i8* [[TWENTY]]
// CHECK-PPC-NEXT:  [[OVERFLOW_AREA1:%[a-z_0-9]+]] = load i8*, i8** [[TWENTYTWO]]
// CHECK-PPC-NEXT:  [[TWENTYFIVE:%[0-9]+]] = ptrtoint i8* [[OVERFLOW_AREA1]] to i32
// CHECK-PPC-NEXT:  [[REGSAVE_AREA1:%[a-z_0-9]+]] = load i8*, i8** [[TWENTYFOUR]]
// CHECK-PPC-NEXT:  [[TWENTYSIX:%[0-9]+]] = ptrtoint i8* [[REGSAVE_AREA1]] to i32
// CHECK-PPC-NEXT:  [[COND1:%[a-z0-9]+]] = icmp ult i8 [[GPR1]], 8
// CHECK-PPC-NEXT:  [[TWENTYSEVEN:%[0-9]+]] = mul i8 [[GPR1]], 4
// CHECK-PPC-NEXT:  [[TWENTYEIGHT:%[0-9]+]] = sext i8 [[TWENTYSEVEN]] to i32
// CHECK-PPC-NEXT:  [[TWENTYNINE:%[0-9]+]] = add i32 [[TWENTYSIX]], [[TWENTYEIGHT]]
// CHECK-PPC-NEXT:  br i1 [[COND1]], label [[USING_REGS1:%[a-z_0-9]+]], label [[USING_OVERFLOW1:%[a-z_0-9]+]]
//
// CHECK-PPC1:[[USING_REGS1]]:
// CHECK-PPC:  [[THIRTY:%[0-9]+]] = inttoptr i32 [[TWENTYNINE]] to i32*
// CHECK-PPC-NEXT:  [[THIRTYONE:%[0-9]+]] = add i8 [[GPR1]], 1
// CHECK-PPC-NEXT:  store i8 [[THIRTYONE]], i8* [[GPRPTR1]]
// CHECK-PPC-NEXT:  br label [[CONT1:%[a-z0-9]+]]
//
// CHECK-PPC1:[[USING_OVERFLOW1]]:
// CHECK-PPC:  [[THIRTYTWO:%[0-9]+]] = inttoptr i32 [[TWENTYFIVE]] to i32*
// CHECK-PPC-NEXT:  [[THIRTYTHREE:%[0-9]+]] = add i32 [[TWENTYFIVE]], 4
// CHECK-PPC-NEXT:  [[THIRTYFOUR:%[0-9]+]] = inttoptr i32 [[THIRTYTHREE]] to i8*
// CHECK-PPC-NEXT:  store i8* [[THIRTYFOUR]], i8** [[TWENTYTWO]]
// CHECK-PPC-NEXT:  br label [[CONT1]]
//
// CHECK-PPC1:[[CONT1]]:
// CHECK-PPC:  [[VAARG_ADDR1:%[a-z.0-9]+]] = phi i32* [ [[THIRTY]], [[USING_REGS1]] ], [ [[THIRTYTWO]], [[USING_OVERFLOW1]] ]
// CHECK-PPC-NEXT:  [[THIRTYFIVE:%[0-9]+]] = load i32, i32* [[VAARG_ADDR1]]
// CHECK-PPC-NEXT:  store i32 [[THIRTYFIVE]], i32* %v, align 4

#ifdef __powerpc64__
  __int128_t u = va_arg (ap, __int128_t);
#endif
// CHECK: bitcast i8* %{{[a-z.0-9]+}} to i128*
// CHECK-NEXT: load i128, i128* %{{[0-9]+}}
}
