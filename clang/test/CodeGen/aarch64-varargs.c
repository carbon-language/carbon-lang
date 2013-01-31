// RUN: %clang_cc1 -triple aarch64 -emit-llvm -o - %s | FileCheck %s
#include <stdarg.h>

// Obviously there's more than one way to implement va_arg. This test should at
// least prevent unintentional regressions caused by refactoring.

va_list the_list;

int simple_int(void) {
// CHECK: define i32 @simple_int
  return va_arg(the_list, int);
// CHECK: [[GR_OFFS:%[a-z_0-9]+]] = load i32* getelementptr inbounds (%struct.__va_list* @the_list, i32 0, i32 3)
// CHECK: [[EARLY_ONSTACK:%[a-z_0-9]+]] = icmp sge i32 [[GR_OFFS]], 0
// CHECK: br i1 [[EARLY_ONSTACK]], label %[[VAARG_ON_STACK:[a-z_.0-9]+]], label %[[VAARG_MAYBE_REG:[a-z_.0-9]+]]

// CHECK: [[VAARG_MAYBE_REG]]
// CHECK: [[NEW_REG_OFFS:%[a-z_0-9]+]] = add i32 [[GR_OFFS]], 8
// CHECK: store i32 [[NEW_REG_OFFS]], i32* getelementptr inbounds (%struct.__va_list* @the_list, i32 0, i32 3)
// CHECK: [[INREG:%[a-z_0-9]+]] = icmp sle i32 [[NEW_REG_OFFS]], 0
// CHECK: br i1 [[INREG]], label %[[VAARG_IN_REG:[a-z_.0-9]+]], label %[[VAARG_ON_STACK]]

// CHECK: [[VAARG_IN_REG]]
// CHECK: [[REG_TOP:%[a-z_0-9]+]] = load i8** getelementptr inbounds (%struct.__va_list* @the_list, i32 0, i32 1)
// CHECK: [[REG_ADDR:%[a-z_0-9]+]] = getelementptr i8* [[REG_TOP]], i32 [[GR_OFFS]]
// CHECK: [[FROMREG_ADDR:%[a-z_0-9]+]] = bitcast i8* [[REG_ADDR]] to i32*
// CHECK: br label %[[VAARG_END:[a-z._0-9]+]]

// CHECK: [[VAARG_ON_STACK]]
// CHECK: [[STACK:%[a-z_0-9]+]] = load i8** getelementptr inbounds (%struct.__va_list* @the_list, i32 0, i32 0)
// CHECK: [[NEW_STACK:%[a-z_0-9]+]] = getelementptr i8* [[STACK]], i32 8
// CHECK: store i8* [[NEW_STACK]], i8** getelementptr inbounds (%struct.__va_list* @the_list, i32 0, i32 0)
// CHECK: [[FROMSTACK_ADDR:%[a-z_0-9]+]] = bitcast i8* [[STACK]] to i32*
// CHECK: br label %[[VAARG_END]]

// CHECK: [[VAARG_END]]
// CHECK: [[ADDR:%[a-z._0-9]+]] = phi i32* [ [[FROMREG_ADDR]], %[[VAARG_IN_REG]] ], [ [[FROMSTACK_ADDR]], %[[VAARG_ON_STACK]] ]
// CHECK: [[RESULT:%[a-z_0-9]+]] = load i32* [[ADDR]]
// CHECK: ret i32 [[RESULT]]
}

__int128 aligned_int(void) {
// CHECK: define i128 @aligned_int
  return va_arg(the_list, __int128);
// CHECK: [[GR_OFFS:%[a-z_0-9]+]] = load i32* getelementptr inbounds (%struct.__va_list* @the_list, i32 0, i32 3)
// CHECK: [[EARLY_ONSTACK:%[a-z_0-9]+]] = icmp sge i32 [[GR_OFFS]], 0
// CHECK: br i1 [[EARLY_ONSTACK]], label %[[VAARG_ON_STACK:[a-z_.0-9]+]], label %[[VAARG_MAYBE_REG:[a-z_.0-9]+]]

// CHECK: [[VAARG_MAYBE_REG]]
// CHECK: [[ALIGN_REGOFFS:%[a-z_0-9]+]] = add i32 [[GR_OFFS]], 15
// CHECK: [[ALIGNED_REGOFFS:%[a-z_0-9]+]] = and i32 [[ALIGN_REGOFFS]], -16
// CHECK: [[NEW_REG_OFFS:%[a-z_0-9]+]] = add i32 [[ALIGNED_REGOFFS]], 16
// CHECK: store i32 [[NEW_REG_OFFS]], i32* getelementptr inbounds (%struct.__va_list* @the_list, i32 0, i32 3)
// CHECK: [[INREG:%[a-z_0-9]+]] = icmp sle i32 [[NEW_REG_OFFS]], 0
// CHECK: br i1 [[INREG]], label %[[VAARG_IN_REG:[a-z_.0-9]+]], label %[[VAARG_ON_STACK]]

// CHECK: [[VAARG_IN_REG]]
// CHECK: [[REG_TOP:%[a-z_0-9]+]] = load i8** getelementptr inbounds (%struct.__va_list* @the_list, i32 0, i32 1)
// CHECK: [[REG_ADDR:%[a-z_0-9]+]] = getelementptr i8* [[REG_TOP]], i32 [[ALIGNED_REGOFFS]]
// CHECK: [[FROMREG_ADDR:%[a-z_0-9]+]] = bitcast i8* [[REG_ADDR]] to i128*
// CHECK: br label %[[VAARG_END:[a-z._0-9]+]]

// CHECK: [[VAARG_ON_STACK]]
// CHECK: [[STACK:%[a-z_0-9]+]] = load i8** getelementptr inbounds (%struct.__va_list* @the_list, i32 0, i32 0)
// CHECK: [[STACKINT:%[a-z_0-9]+]] = ptrtoint i8* [[STACK]] to i64
// CHECK: [[ALIGN_STACK:%[a-z_0-9]+]] = add i64 [[STACKINT]], 15
// CHECK: [[ALIGNED_STACK_INT:%[a-z_0-9]+]] = and i64 [[ALIGN_STACK]], -16
// CHECK: [[ALIGNED_STACK_PTR:%[a-z_0-9]+]] = inttoptr i64 [[ALIGNED_STACK_INT]] to i8*
// CHECK: [[NEW_STACK:%[a-z_0-9]+]] = getelementptr i8* [[ALIGNED_STACK_PTR]], i32 16
// CHECK: store i8* [[NEW_STACK]], i8** getelementptr inbounds (%struct.__va_list* @the_list, i32 0, i32 0)
// CHECK: [[FROMSTACK_ADDR:%[a-z_0-9]+]] = bitcast i8* [[ALIGNED_STACK_PTR]] to i128*
// CHECK: br label %[[VAARG_END]]

// CHECK: [[VAARG_END]]
// CHECK: [[ADDR:%[a-z._0-9]+]] = phi i128* [ [[FROMREG_ADDR]], %[[VAARG_IN_REG]] ], [ [[FROMSTACK_ADDR]], %[[VAARG_ON_STACK]] ]
// CHECK: [[RESULT:%[a-z_0-9]+]] = load i128* [[ADDR]]
// CHECK: ret i128 [[RESULT]]
}

struct bigstruct {
  int a[10];
};

struct bigstruct simple_indirect(void) {
// CHECK: define void @simple_indirect
  return va_arg(the_list, struct bigstruct);
// CHECK: [[GR_OFFS:%[a-z_0-9]+]] = load i32* getelementptr inbounds (%struct.__va_list* @the_list, i32 0, i32 3)
// CHECK: [[EARLY_ONSTACK:%[a-z_0-9]+]] = icmp sge i32 [[GR_OFFS]], 0
// CHECK: br i1 [[EARLY_ONSTACK]], label %[[VAARG_ON_STACK:[a-z_.0-9]+]], label %[[VAARG_MAYBE_REG:[a-z_.0-9]+]]

// CHECK: [[VAARG_MAYBE_REG]]
// CHECK-NOT: and i32
// CHECK: [[NEW_REG_OFFS:%[a-z_0-9]+]] = add i32 [[GR_OFFS]], 8
// CHECK: store i32 [[NEW_REG_OFFS]], i32* getelementptr inbounds (%struct.__va_list* @the_list, i32 0, i32 3)
// CHECK: [[INREG:%[a-z_0-9]+]] = icmp sle i32 [[NEW_REG_OFFS]], 0
// CHECK: br i1 [[INREG]], label %[[VAARG_IN_REG:[a-z_.0-9]+]], label %[[VAARG_ON_STACK]]

// CHECK: [[VAARG_IN_REG]]
// CHECK: [[REG_TOP:%[a-z_0-9]+]] = load i8** getelementptr inbounds (%struct.__va_list* @the_list, i32 0, i32 1)
// CHECK: [[REG_ADDR:%[a-z_0-9]+]] = getelementptr i8* [[REG_TOP]], i32 [[GR_OFFS]]
// CHECK: [[FROMREG_ADDR:%[a-z_0-9]+]] = bitcast i8* [[REG_ADDR]] to %struct.bigstruct**
// CHECK: br label %[[VAARG_END:[a-z._0-9]+]]

// CHECK: [[VAARG_ON_STACK]]
// CHECK: [[STACK:%[a-z_0-9]+]] = load i8** getelementptr inbounds (%struct.__va_list* @the_list, i32 0, i32 0)
// CHECK-NOT: and i64
// CHECK: [[NEW_STACK:%[a-z_0-9]+]] = getelementptr i8* [[STACK]], i32 8
// CHECK: store i8* [[NEW_STACK]], i8** getelementptr inbounds (%struct.__va_list* @the_list, i32 0, i32 0)
// CHECK: [[FROMSTACK_ADDR:%[a-z_0-9]+]] = bitcast i8* [[STACK]] to %struct.bigstruct**
// CHECK: br label %[[VAARG_END]]

// CHECK: [[VAARG_END]]
// CHECK: [[ADDR:%[a-z._0-9]+]] = phi %struct.bigstruct** [ [[FROMREG_ADDR]], %[[VAARG_IN_REG]] ], [ [[FROMSTACK_ADDR]], %[[VAARG_ON_STACK]] ]
// CHECK: load %struct.bigstruct** [[ADDR]]
}

struct aligned_bigstruct {
  float a;
  long double b;
};

struct aligned_bigstruct simple_aligned_indirect(void) {
// CHECK: define void @simple_aligned_indirect
  return va_arg(the_list, struct aligned_bigstruct);
// CHECK: [[GR_OFFS:%[a-z_0-9]+]] = load i32* getelementptr inbounds (%struct.__va_list* @the_list, i32 0, i32 3)
// CHECK: [[EARLY_ONSTACK:%[a-z_0-9]+]] = icmp sge i32 [[GR_OFFS]], 0
// CHECK: br i1 [[EARLY_ONSTACK]], label %[[VAARG_ON_STACK:[a-z_.0-9]+]], label %[[VAARG_MAYBE_REG:[a-z_.0-9]+]]

// CHECK: [[VAARG_MAYBE_REG]]
// CHECK: [[NEW_REG_OFFS:%[a-z_0-9]+]] = add i32 [[GR_OFFS]], 8
// CHECK: store i32 [[NEW_REG_OFFS]], i32* getelementptr inbounds (%struct.__va_list* @the_list, i32 0, i32 3)
// CHECK: [[INREG:%[a-z_0-9]+]] = icmp sle i32 [[NEW_REG_OFFS]], 0
// CHECK: br i1 [[INREG]], label %[[VAARG_IN_REG:[a-z_.0-9]+]], label %[[VAARG_ON_STACK]]

// CHECK: [[VAARG_IN_REG]]
// CHECK: [[REG_TOP:%[a-z_0-9]+]] = load i8** getelementptr inbounds (%struct.__va_list* @the_list, i32 0, i32 1)
// CHECK: [[REG_ADDR:%[a-z_0-9]+]] = getelementptr i8* [[REG_TOP]], i32 [[GR_OFFS]]
// CHECK: [[FROMREG_ADDR:%[a-z_0-9]+]] = bitcast i8* [[REG_ADDR]] to %struct.aligned_bigstruct**
// CHECK: br label %[[VAARG_END:[a-z._0-9]+]]

// CHECK: [[VAARG_ON_STACK]]
// CHECK: [[STACK:%[a-z_0-9]+]] = load i8** getelementptr inbounds (%struct.__va_list* @the_list, i32 0, i32 0)
// CHECK: [[NEW_STACK:%[a-z_0-9]+]] = getelementptr i8* [[STACK]], i32 8
// CHECK: store i8* [[NEW_STACK]], i8** getelementptr inbounds (%struct.__va_list* @the_list, i32 0, i32 0)
// CHECK: [[FROMSTACK_ADDR:%[a-z_0-9]+]] = bitcast i8* [[STACK]] to %struct.aligned_bigstruct**
// CHECK: br label %[[VAARG_END]]

// CHECK: [[VAARG_END]]
// CHECK: [[ADDR:%[a-z._0-9]+]] = phi %struct.aligned_bigstruct** [ [[FROMREG_ADDR]], %[[VAARG_IN_REG]] ], [ [[FROMSTACK_ADDR]], %[[VAARG_ON_STACK]] ]
// CHECK: load %struct.aligned_bigstruct** [[ADDR]]
}

double simple_double(void) {
// CHECK: define double @simple_double
  return va_arg(the_list, double);
// CHECK: [[VR_OFFS:%[a-z_0-9]+]] = load i32* getelementptr inbounds (%struct.__va_list* @the_list, i32 0, i32 4)
// CHECK: [[EARLY_ONSTACK:%[a-z_0-9]+]] = icmp sge i32 [[VR_OFFS]], 0
// CHECK: br i1 [[EARLY_ONSTACK]], label %[[VAARG_ON_STACK]], label %[[VAARG_MAYBE_REG]]

// CHECK: [[VAARG_MAYBE_REG]]
// CHECK: [[NEW_REG_OFFS:%[a-z_0-9]+]] = add i32 [[VR_OFFS]], 16
// CHECK: store i32 [[NEW_REG_OFFS]], i32* getelementptr inbounds (%struct.__va_list* @the_list, i32 0, i32 4)
// CHECK: [[INREG:%[a-z_0-9]+]] = icmp sle i32 [[NEW_REG_OFFS]], 0
// CHECK: br i1 [[INREG]], label %[[VAARG_IN_REG:[a-z_.0-9]+]], label %[[VAARG_ON_STACK]]

// CHECK: [[VAARG_IN_REG]]
// CHECK: [[REG_TOP:%[a-z_0-9]+]] = load i8** getelementptr inbounds (%struct.__va_list* @the_list, i32 0, i32 2)
// CHECK: [[REG_ADDR:%[a-z_0-9]+]] = getelementptr i8* [[REG_TOP]], i32 [[VR_OFFS]]
// CHECK: [[FROMREG_ADDR:%[a-z_0-9]+]] = bitcast i8* [[REG_ADDR]] to double*
// CHECK: br label %[[VAARG_END]]

// CHECK: [[VAARG_ON_STACK]]
// CHECK: [[STACK:%[a-z_0-9]+]] = load i8** getelementptr inbounds (%struct.__va_list* @the_list, i32 0, i32 0)
// CHECK: [[NEW_STACK:%[a-z_0-9]+]] = getelementptr i8* [[STACK]], i32 8
// CHECK: store i8* [[NEW_STACK]], i8** getelementptr inbounds (%struct.__va_list* @the_list, i32 0, i32 0)
// CHECK: [[FROMSTACK_ADDR:%[a-z_0-9]+]] = bitcast i8* [[STACK]] to double*
// CHECK: br label %[[VAARG_END]]

// CHECK: [[VAARG_END]]
// CHECK: [[ADDR:%[a-z._0-9]+]] = phi double* [ [[FROMREG_ADDR]], %[[VAARG_IN_REG]] ], [ [[FROMSTACK_ADDR]], %[[VAARG_ON_STACK]] ]
// CHECK: [[RESULT:%[a-z_0-9]+]] = load double* [[ADDR]]
// CHECK: ret double [[RESULT]]
}

struct hfa {
  float a, b;
};

struct hfa simple_hfa(void) {
// CHECK: define %struct.hfa @simple_hfa
  return va_arg(the_list, struct hfa);
// CHECK: [[VR_OFFS:%[a-z_0-9]+]] = load i32* getelementptr inbounds (%struct.__va_list* @the_list, i32 0, i32 4)
// CHECK: [[EARLY_ONSTACK:%[a-z_0-9]+]] = icmp sge i32 [[VR_OFFS]], 0
// CHECK: br i1 [[EARLY_ONSTACK]], label %[[VAARG_ON_STACK:[a-z_.0-9]+]], label %[[VAARG_MAYBE_REG:[a-z_.0-9]+]]

// CHECK: [[VAARG_MAYBE_REG]]
// CHECK: [[NEW_REG_OFFS:%[a-z_0-9]+]] = add i32 [[VR_OFFS]], 32
// CHECK: store i32 [[NEW_REG_OFFS]], i32* getelementptr inbounds (%struct.__va_list* @the_list, i32 0, i32 4)
// CHECK: [[INREG:%[a-z_0-9]+]] = icmp sle i32 [[NEW_REG_OFFS]], 0
// CHECK: br i1 [[INREG]], label %[[VAARG_IN_REG:[a-z_.0-9]+]], label %[[VAARG_ON_STACK]]

// CHECK: [[VAARG_IN_REG]]
// CHECK: [[REG_TOP:%[a-z_0-9]+]] = load i8** getelementptr inbounds (%struct.__va_list* @the_list, i32 0, i32 2)
// CHECK: [[FIRST_REG:%[a-z_0-9]+]] = getelementptr i8* [[REG_TOP]], i32 [[VR_OFFS]]
// CHECK: [[EL_ADDR:%[a-z_0-9]+]] = getelementptr i8* [[FIRST_REG]], i32 0
// CHECK: [[EL_TYPED:%[a-z_0-9]+]] = bitcast i8* [[EL_ADDR]] to float*
// CHECK: [[EL_TMPADDR:%[a-z_0-9]+]] = getelementptr inbounds [2 x float]* %[[TMP_HFA:[a-z_.0-9]+]], i32 0, i32 0
// CHECK: [[EL:%[a-z_0-9]+]] = load float* [[EL_TYPED]]
// CHECK: store float [[EL]], float* [[EL_TMPADDR]]
// CHECK: [[EL_ADDR:%[a-z_0-9]+]] = getelementptr i8* [[FIRST_REG]], i32 16
// CHECK: [[EL_TYPED:%[a-z_0-9]+]] = bitcast i8* [[EL_ADDR]] to float*
// CHECK: [[EL_TMPADDR:%[a-z_0-9]+]] = getelementptr inbounds [2 x float]* %[[TMP_HFA]], i32 0, i32 1
// CHECK: [[EL:%[a-z_0-9]+]] = load float* [[EL_TYPED]]
// CHECK: store float [[EL]], float* [[EL_TMPADDR]]
// CHECK: [[FROMREG_ADDR:%[a-z_0-9]+]] = bitcast [2 x float]* %[[TMP_HFA]] to %struct.hfa*
// CHECK: br label %[[VAARG_END:[a-z_.0-9]+]]

// CHECK: [[VAARG_ON_STACK]]
// CHECK: [[STACK:%[a-z_0-9]+]] = load i8** getelementptr inbounds (%struct.__va_list* @the_list, i32 0, i32 0)
// CHECK: [[NEW_STACK:%[a-z_0-9]+]] = getelementptr i8* [[STACK]], i32 8
// CHECK: store i8* [[NEW_STACK]], i8** getelementptr inbounds (%struct.__va_list* @the_list, i32 0, i32 0)
// CHECK: [[FROMSTACK_ADDR:%[a-z_0-9]+]] = bitcast i8* [[STACK]] to %struct.hfa*
// CHECK: br label %[[VAARG_END]]

// CHECK: [[VAARG_END]]
// CHECK: [[ADDR:%[a-z._0-9]+]] = phi %struct.hfa* [ [[FROMREG_ADDR]], %[[VAARG_IN_REG]] ], [ [[FROMSTACK_ADDR]], %[[VAARG_ON_STACK]] ]
}

void check_start(int n, ...) {
// CHECK: define void @check_start(i32 %n, ...)

  va_list the_list;
  va_start(the_list, n);
// CHECK: [[THE_LIST:%[a-z_0-9]+]] = alloca %struct.__va_list
// CHECK: [[VOIDP_THE_LIST:%[a-z_0-9]+]] = bitcast %struct.__va_list* [[THE_LIST]] to i8*
// CHECK: call void @llvm.va_start(i8* [[VOIDP_THE_LIST]])
}


