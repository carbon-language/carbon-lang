! Test allocatable assignments
! RUN: bbc -emit-fir %s -o - | FileCheck %s

module alloc_assign
  type t
    integer :: i
  end type
contains

! -----------------------------------------------------------------------------
!            Test simple scalar RHS
! -----------------------------------------------------------------------------

! CHECK-LABEL: func @_QMalloc_assignPtest_simple_scalar(
! CHECK-SAME: %[[VAL_0:.*]]: !fir.ref<!fir.box<!fir.heap<f32>>>{{.*}}) {
subroutine test_simple_scalar(x)
  real, allocatable  :: x
! CHECK:  %[[VAL_1:.*]] = arith.constant 4.200000e+01 : f32
! CHECK:  %[[VAL_2:.*]] = fir.load %[[VAL_0]] : !fir.ref<!fir.box<!fir.heap<f32>>>
! CHECK:  %[[VAL_3:.*]] = fir.box_addr %[[VAL_2]] : (!fir.box<!fir.heap<f32>>) -> !fir.heap<f32>
! CHECK:  %[[VAL_4:.*]] = fir.convert %[[VAL_3]] : (!fir.heap<f32>) -> i64
! CHECK:  %[[VAL_5:.*]] = arith.constant 0 : i64
! CHECK:  %[[VAL_6:.*]] = arith.cmpi ne, %[[VAL_4]], %[[VAL_5]] : i64
! CHECK:  %[[VAL_7:.*]]:2 = fir.if %[[VAL_6]] -> (i1, !fir.heap<f32>) {
! CHECK:    %[[VAL_8:.*]] = arith.constant false
! CHECK:    %[[VAL_9:.*]] = fir.if %[[VAL_8]] -> (!fir.heap<f32>) {
! CHECK:      %[[VAL_10:.*]] = fir.allocmem f32 {uniq_name = ".auto.alloc"}
! CHECK:      fir.result %[[VAL_10]] : !fir.heap<f32>
! CHECK:    } else {
! CHECK:      fir.result %[[VAL_3]] : !fir.heap<f32>
! CHECK:    }
! CHECK:    fir.result %[[VAL_8]], %[[VAL_11:.*]] : i1, !fir.heap<f32>
! CHECK:  } else {
! CHECK:    %[[VAL_12:.*]] = arith.constant true
! CHECK:    %[[VAL_13:.*]] = fir.allocmem f32 {uniq_name = ".auto.alloc"}
! CHECK:    fir.result %[[VAL_12]], %[[VAL_13]] : i1, !fir.heap<f32>
! CHECK:  }
! CHECK:  fir.store %[[VAL_1]] to %[[VAL_14:.*]]#1 : !fir.heap<f32>
! CHECK:  fir.if %[[VAL_14]]#0 {
! CHECK:    fir.if %[[VAL_6]] {
! CHECK:      fir.freemem %[[VAL_3]]
! CHECK:    }
! CHECK:    %[[VAL_15:.*]] = fir.embox %[[VAL_14]]#1 : (!fir.heap<f32>) -> !fir.box<!fir.heap<f32>>
! CHECK:    fir.store %[[VAL_15]] to %[[VAL_0]] : !fir.ref<!fir.box<!fir.heap<f32>>>
! CHECK:  }
  x = 42.
end subroutine

! CHECK-LABEL: func @_QMalloc_assignPtest_simple_local_scalar() {
subroutine test_simple_local_scalar()
  real, allocatable  :: x
! CHECK:  %[[VAL_1:.*]] = fir.alloca !fir.heap<f32> {uniq_name = "_QMalloc_assignFtest_simple_local_scalarEx.addr"}
! CHECK:  %[[VAL_2:.*]] = fir.zero_bits !fir.heap<f32>
! CHECK:  fir.store %[[VAL_2]] to %[[VAL_1]] : !fir.ref<!fir.heap<f32>>
! CHECK:  %[[VAL_3:.*]] = arith.constant 4.200000e+01 : f32
! CHECK:  %[[VAL_4:.*]] = fir.load %[[VAL_1]] : !fir.ref<!fir.heap<f32>>
! CHECK:  %[[VAL_5:.*]] = fir.convert %[[VAL_4]] : (!fir.heap<f32>) -> i64
! CHECK:  %[[VAL_6:.*]] = arith.constant 0 : i64
! CHECK:  %[[VAL_7:.*]] = arith.cmpi ne, %[[VAL_5]], %[[VAL_6]] : i64
! CHECK:  %[[VAL_8:.*]]:2 = fir.if %[[VAL_7]] -> (i1, !fir.heap<f32>) {
! CHECK:    %[[VAL_9:.*]] = arith.constant false
! CHECK:    %[[VAL_10:.*]] = fir.if %[[VAL_9]] -> (!fir.heap<f32>) {
! CHECK:      %[[VAL_11:.*]] = fir.allocmem f32 {uniq_name = ".auto.alloc"}
! CHECK:      fir.result %[[VAL_11]] : !fir.heap<f32>
! CHECK:    } else {
! CHECK:      fir.result %[[VAL_4]] : !fir.heap<f32>
! CHECK:    }
! CHECK:    fir.result %[[VAL_9]], %[[VAL_12:.*]] : i1, !fir.heap<f32>
! CHECK:  } else {
! CHECK:    %[[VAL_13:.*]] = arith.constant true
! CHECK:    %[[VAL_14:.*]] = fir.allocmem f32 {uniq_name = ".auto.alloc"}
! CHECK:    fir.result %[[VAL_13]], %[[VAL_14]] : i1, !fir.heap<f32>
! CHECK:  }
! CHECK:  fir.store %[[VAL_3]] to %[[VAL_15:.*]]#1 : !fir.heap<f32>
! CHECK:  fir.if %[[VAL_15]]#0 {
! CHECK:    fir.if %[[VAL_7]] {
! CHECK:      fir.freemem %[[VAL_4]]
! CHECK:    }
! CHECK:    fir.store %[[VAL_15]]#1 to %[[VAL_1]] : !fir.ref<!fir.heap<f32>>
! CHECK:  }
  x = 42.
end subroutine

! -----------------------------------------------------------------------------
!            Test character scalar RHS
! -----------------------------------------------------------------------------

! CHECK-LABEL: func @_QMalloc_assignPtest_deferred_char_scalar(
! CHECK-SAME:  %[[VAL_0:.*]]: !fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>>{{.*}}) {
subroutine test_deferred_char_scalar(x)
  character(:), allocatable  :: x
! CHECK:  %[[VAL_1:.*]] = fir.address_of(@_QQ{{.*}}) : !fir.ref<!fir.char<1,12>>
! CHECK:  %[[VAL_2:.*]] = arith.constant 12 : index
! CHECK:  %[[VAL_3:.*]] = fir.load %[[VAL_0]] : !fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>>
! CHECK:  %[[VAL_4:.*]] = fir.box_addr %[[VAL_3]] : (!fir.box<!fir.heap<!fir.char<1,?>>>) -> !fir.heap<!fir.char<1,?>>
! CHECK:  %[[VAL_5:.*]] = fir.convert %[[VAL_4]] : (!fir.heap<!fir.char<1,?>>) -> i64
! CHECK:  %[[VAL_6:.*]] = arith.constant 0 : i64
! CHECK:  %[[VAL_7:.*]] = arith.cmpi ne, %[[VAL_5]], %[[VAL_6]] : i64
! CHECK:  %[[VAL_8:.*]]:2 = fir.if %[[VAL_7]] -> (i1, !fir.heap<!fir.char<1,?>>) {
! CHECK:    %[[VAL_9:.*]] = arith.constant false
! CHECK:    %[[VAL_10:.*]] = fir.box_elesize %[[VAL_3]] : (!fir.box<!fir.heap<!fir.char<1,?>>>) -> index
! CHECK:    %[[VAL_11:.*]] = arith.cmpi ne, %[[VAL_10]], %[[VAL_2]] : index
! CHECK:    %[[VAL_12:.*]] = arith.select %[[VAL_11]], %[[VAL_11]], %[[VAL_9]] : i1
! CHECK:    %[[VAL_13:.*]] = fir.if %[[VAL_12]] -> (!fir.heap<!fir.char<1,?>>) {
! CHECK:      %[[VAL_14:.*]] = fir.allocmem !fir.char<1,?>(%[[VAL_2]] : index) {uniq_name = ".auto.alloc"}
! CHECK:      fir.result %[[VAL_14]] : !fir.heap<!fir.char<1,?>>
! CHECK:    } else {
! CHECK:      fir.result %[[VAL_4]] : !fir.heap<!fir.char<1,?>>
! CHECK:    }
! CHECK:    fir.result %[[VAL_12]], %[[VAL_15:.*]] : i1, !fir.heap<!fir.char<1,?>>
! CHECK:  } else {
! CHECK:    %[[VAL_16:.*]] = arith.constant true
! CHECK:    %[[VAL_17:.*]] = fir.allocmem !fir.char<1,?>(%[[VAL_2]] : index) {uniq_name = ".auto.alloc"}
! CHECK:    fir.result %[[VAL_16]], %[[VAL_17]] : i1, !fir.heap<!fir.char<1,?>>
! CHECK:  }

! character assignment ...
! CHECK:  %[[VAL_24:.*]] = fir.convert %[[VAL_8]]#1 : (!fir.heap<!fir.char<1,?>>) -> !fir.ref<i8>
! CHECK:  fir.call @llvm.memmove.p0.p0.i64(%[[VAL_24]], %{{.*}}, %{{.*}}, %{{.*}}) : (!fir.ref<i8>, !fir.ref<i8>, i64, i1) -> ()
! character assignment ...

! CHECK:  fir.if %[[VAL_8]]#0 {
! CHECK:    fir.if %[[VAL_7]] {
! CHECK:      fir.freemem %[[VAL_4]]
! CHECK:    }
! CHECK:    %[[VAL_36:.*]] = fir.embox %[[VAL_8]]#1 typeparams %[[VAL_2]] : (!fir.heap<!fir.char<1,?>>, index) -> !fir.box<!fir.heap<!fir.char<1,?>>>
! CHECK:    fir.store %[[VAL_36]] to %[[VAL_0]] : !fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>>
! CHECK:  }
  x = "Hello world!"
end subroutine

! CHECK-LABEL: func @_QMalloc_assignPtest_cst_char_scalar(
! CHECK-SAME:  %[[VAL_0:.*]]: !fir.ref<!fir.box<!fir.heap<!fir.char<1,10>>>>{{.*}}) {
subroutine test_cst_char_scalar(x)
  character(10), allocatable  :: x
! CHECK:  %[[VAL_1:.*]] = arith.constant 10 : index
! CHECK:  %[[VAL_2:.*]] = fir.address_of(@_QQ{{.*}}) : !fir.ref<!fir.char<1,12>>
! CHECK:  %[[VAL_3:.*]] = arith.constant 12 : index
! CHECK:  %[[VAL_4:.*]] = fir.load %[[VAL_0]] : !fir.ref<!fir.box<!fir.heap<!fir.char<1,10>>>>
! CHECK:  %[[VAL_5:.*]] = fir.box_addr %[[VAL_4]] : (!fir.box<!fir.heap<!fir.char<1,10>>>) -> !fir.heap<!fir.char<1,10>>
! CHECK:  %[[VAL_6:.*]] = fir.convert %[[VAL_5]] : (!fir.heap<!fir.char<1,10>>) -> i64
! CHECK:  %[[VAL_7:.*]] = arith.constant 0 : i64
! CHECK:  %[[VAL_8:.*]] = arith.cmpi ne, %[[VAL_6]], %[[VAL_7]] : i64
! CHECK:  %[[VAL_9:.*]]:2 = fir.if %[[VAL_8]] -> (i1, !fir.heap<!fir.char<1,10>>) {
! CHECK:    %[[VAL_10:.*]] = arith.constant false
! CHECK:    %[[VAL_11:.*]] = fir.if %[[VAL_10]] -> (!fir.heap<!fir.char<1,10>>) {
! CHECK:      %[[VAL_12:.*]] = fir.allocmem !fir.char<1,10> {uniq_name = ".auto.alloc"}
! CHECK:      fir.result %[[VAL_12]] : !fir.heap<!fir.char<1,10>>
! CHECK:    } else {
! CHECK:      fir.result %[[VAL_5]] : !fir.heap<!fir.char<1,10>>
! CHECK:    }
! CHECK:    fir.result %[[VAL_10]], %[[VAL_13:.*]] : i1, !fir.heap<!fir.char<1,10>>
! CHECK:  } else {
! CHECK:    %[[VAL_14:.*]] = arith.constant true
! CHECK:    %[[VAL_15:.*]] = fir.allocmem !fir.char<1,10> {uniq_name = ".auto.alloc"}
! CHECK:    fir.result %[[VAL_14]], %[[VAL_15]] : i1, !fir.heap<!fir.char<1,10>>
! CHECK:  }

! character assignment ...
! CHECK:  %[[VAL_24:.*]] = fir.convert %[[VAL_9]]#1 : (!fir.heap<!fir.char<1,10>>) -> !fir.ref<i8>
! CHECK:  fir.call @llvm.memmove.p0.p0.i64(%[[VAL_24]], %{{.*}}, %{{.*}}, %{{.*}}) : (!fir.ref<i8>, !fir.ref<i8>, i64, i1) -> ()
! character assignment ...

! CHECK:  fir.if %[[VAL_9]]#0 {
! CHECK:    fir.if %[[VAL_8]] {
! CHECK:      fir.freemem %[[VAL_5]]
! CHECK:    }
! CHECK:    %[[VAL_34:.*]] = fir.embox %[[VAL_9]]#1 : (!fir.heap<!fir.char<1,10>>) -> !fir.box<!fir.heap<!fir.char<1,10>>>
! CHECK:    fir.store %[[VAL_34]] to %[[VAL_0]] : !fir.ref<!fir.box<!fir.heap<!fir.char<1,10>>>>
! CHECK:  }
  x = "Hello world!"
end subroutine

! CHECK-LABEL: func @_QMalloc_assignPtest_dyn_char_scalar(
! CHECK-SAME:  %[[VAL_0:.*]]: !fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>>{{.*}},
! CHECK-SAME:  %[[VAL_1:.*]]: !fir.ref<i32>{{.*}}) {
subroutine test_dyn_char_scalar(x, n)
  integer :: n
  character(n), allocatable  :: x
! CHECK:  %[[VAL_2A:.*]] = fir.load %[[VAL_1]] : !fir.ref<i32>
! CHECK:  %[[c0_i32:.*]] = arith.constant 0 : i32
! CHECK:  %[[VAL_2B:.*]] = arith.cmpi sgt, %[[VAL_2A]], %[[c0_i32]] : i32
! CHECK:  %[[VAL_2:.*]] = arith.select %[[VAL_2B]], %[[VAL_2A]], %[[c0_i32]] : i32
! CHECK:  %[[VAL_3:.*]] = fir.address_of(@_QQcl.48656C6C6F20776F726C6421) : !fir.ref<!fir.char<1,12>>
! CHECK:  %[[VAL_4:.*]] = arith.constant 12 : index
! CHECK:  %[[VAL_5:.*]] = fir.load %[[VAL_0]] : !fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>>
! CHECK:  %[[VAL_6:.*]] = fir.box_addr %[[VAL_5]] : (!fir.box<!fir.heap<!fir.char<1,?>>>) -> !fir.heap<!fir.char<1,?>>
! CHECK:  %[[VAL_7:.*]] = fir.convert %[[VAL_6]] : (!fir.heap<!fir.char<1,?>>) -> i64
! CHECK:  %[[VAL_8:.*]] = arith.constant 0 : i64
! CHECK:  %[[VAL_9:.*]] = arith.cmpi ne, %[[VAL_7]], %[[VAL_8]] : i64
! CHECK:  %[[VAL_10:.*]]:2 = fir.if %[[VAL_9]] -> (i1, !fir.heap<!fir.char<1,?>>) {
! CHECK:    %[[VAL_11:.*]] = arith.constant false
! CHECK:    %[[VAL_12:.*]] = fir.if %[[VAL_11]] -> (!fir.heap<!fir.char<1,?>>) {
! CHECK:      %[[VAL_13:.*]] = fir.convert %[[VAL_2]] : (i32) -> index
! CHECK:      %[[VAL_14:.*]] = fir.allocmem !fir.char<1,?>(%[[VAL_13]] : index) {uniq_name = ".auto.alloc"}
! CHECK:      fir.result %[[VAL_14]] : !fir.heap<!fir.char<1,?>>
! CHECK:    } else {
! CHECK:      fir.result %[[VAL_6]] : !fir.heap<!fir.char<1,?>>
! CHECK:    }
! CHECK:    fir.result %[[VAL_11]], %[[VAL_15:.*]] : i1, !fir.heap<!fir.char<1,?>>
! CHECK:  } else {
! CHECK:    %[[VAL_16:.*]] = arith.constant true
! CHECK:    %[[VAL_17:.*]] = fir.convert %[[VAL_2]] : (i32) -> index
! CHECK:    %[[VAL_18:.*]] = fir.allocmem !fir.char<1,?>(%[[VAL_17]] : index) {uniq_name = ".auto.alloc"}
! CHECK:    fir.result %[[VAL_16]], %[[VAL_18]] : i1, !fir.heap<!fir.char<1,?>>
! CHECK:  }

! character assignment ...
! CHECK:  %[[VAL_24:.*]] = fir.convert %[[VAL_10]]#1 : (!fir.heap<!fir.char<1,?>>) -> !fir.ref<i8>
! CHECK:  fir.call @llvm.memmove.p0.p0.i64(%[[VAL_24]], %{{.*}}, %{{.*}}, %{{.*}}) : (!fir.ref<i8>, !fir.ref<i8>, i64, i1) -> ()
! character assignment ...

! CHECK:  fir.if %[[VAL_10]]#0 {
! CHECK:    %[[VAL_39:.*]] = fir.convert %[[VAL_2]] : (i32) -> index
! CHECK:    fir.if %[[VAL_9]] {
! CHECK:      fir.freemem %[[VAL_6]]
! CHECK:    }
! CHECK:    %[[VAL_40:.*]] = fir.embox %[[VAL_10]]#1 typeparams %[[VAL_39]] : (!fir.heap<!fir.char<1,?>>, index) -> !fir.box<!fir.heap<!fir.char<1,?>>>
! CHECK:    fir.store %[[VAL_40]] to %[[VAL_0]] : !fir.ref<!fir.box<!fir.heap<!fir.char<1,?>>>>
! CHECK:  }
  x = "Hello world!"
end subroutine

! CHECK-LABEL: func @_QMalloc_assignPtest_derived_scalar(
! CHECK-SAME:  %[[VAL_0:.*]]: !fir.ref<!fir.box<!fir.heap<!fir.type<_QMalloc_assignTt{i:i32}>>>>{{.*}},
! CHECK-SAME:  %[[VAL_1:.*]]: !fir.ref<!fir.type<_QMalloc_assignTt{i:i32}>>{{.*}}) {
subroutine test_derived_scalar(x, s)
  type(t), allocatable  :: x
  type(t) :: s
  x = s
! CHECK:  %[[VAL_2:.*]] = fir.load %[[VAL_0]] : !fir.ref<!fir.box<!fir.heap<!fir.type<_QMalloc_assignTt{i:i32}>>>>
! CHECK:  %[[VAL_3:.*]] = fir.box_addr %[[VAL_2]] : (!fir.box<!fir.heap<!fir.type<_QMalloc_assignTt{i:i32}>>>) -> !fir.heap<!fir.type<_QMalloc_assignTt{i:i32}>>
! CHECK:  %[[VAL_4:.*]] = fir.convert %[[VAL_3]] : (!fir.heap<!fir.type<_QMalloc_assignTt{i:i32}>>) -> i64
! CHECK:  %[[VAL_5:.*]] = arith.constant 0 : i64
! CHECK:  %[[VAL_6:.*]] = arith.cmpi ne, %[[VAL_4]], %[[VAL_5]] : i64
! CHECK:  %[[VAL_7:.*]]:2 = fir.if %[[VAL_6]] -> (i1, !fir.heap<!fir.type<_QMalloc_assignTt{i:i32}>>) {
! CHECK:    %[[VAL_8:.*]] = arith.constant false
! CHECK:    %[[VAL_9:.*]] = fir.if %[[VAL_8]] -> (!fir.heap<!fir.type<_QMalloc_assignTt{i:i32}>>) {
! CHECK:      %[[VAL_10:.*]] = fir.allocmem !fir.type<_QMalloc_assignTt{i:i32}> {uniq_name = ".auto.alloc"}
! CHECK:      fir.result %[[VAL_10]] : !fir.heap<!fir.type<_QMalloc_assignTt{i:i32}>>
! CHECK:    } else {
! CHECK:      fir.result %[[VAL_3]] : !fir.heap<!fir.type<_QMalloc_assignTt{i:i32}>>
! CHECK:    }
! CHECK:    fir.result %[[VAL_8]], %[[VAL_11:.*]] : i1, !fir.heap<!fir.type<_QMalloc_assignTt{i:i32}>>
! CHECK:  } else {
! CHECK:    %[[VAL_12:.*]] = arith.constant true
! CHECK:    %[[VAL_13:.*]] = fir.allocmem !fir.type<_QMalloc_assignTt{i:i32}> {uniq_name = ".auto.alloc"}
! CHECK:    fir.result %[[VAL_12]], %[[VAL_13]] : i1, !fir.heap<!fir.type<_QMalloc_assignTt{i:i32}>>
! CHECK:  }
! CHECK:  %[[VAL_14:.*]] = fir.field_index i, !fir.type<_QMalloc_assignTt{i:i32}>
! CHECK:  %[[VAL_15:.*]] = fir.coordinate_of %[[VAL_1]], %[[VAL_14]] : (!fir.ref<!fir.type<_QMalloc_assignTt{i:i32}>>, !fir.field) -> !fir.ref<i32>
! CHECK:  %[[VAL_14b:.*]] = fir.field_index i, !fir.type<_QMalloc_assignTt{i:i32}>
! CHECK:  %[[VAL_16:.*]] = fir.coordinate_of %[[VAL_7]]#1, %[[VAL_14b]] : (!fir.heap<!fir.type<_QMalloc_assignTt{i:i32}>>, !fir.field) -> !fir.ref<i32>
! CHECK:  %[[VAL_17:.*]] = fir.load %[[VAL_15]] : !fir.ref<i32>
! CHECK:  fir.store %[[VAL_17]] to %[[VAL_16]] : !fir.ref<i32
! CHECK:  fir.if %[[VAL_7]]#0 {
! CHECK:    fir.if %[[VAL_6]] {
! CHECK:      fir.freemem %[[VAL_3]]
! CHECK:    }
! CHECK:    %[[VAL_19:.*]] = fir.embox %[[VAL_7]]#1 : (!fir.heap<!fir.type<_QMalloc_assignTt{i:i32}>>) -> !fir.box<!fir.heap<!fir.type<_QMalloc_assignTt{i:i32}>>>
! CHECK:    fir.store %[[VAL_19]] to %[[VAL_0]] : !fir.ref<!fir.box<!fir.heap<!fir.type<_QMalloc_assignTt{i:i32}>>>>
! CHECK:  }
end subroutine

! -----------------------------------------------------------------------------
!            Test numeric/logical array RHS
! -----------------------------------------------------------------------------

! CHECK-LABEL: func @_QMalloc_assignPtest_from_cst_shape_array(
! CHECK-SAME:  %[[VAL_0:.*]]: !fir.ref<!fir.box<!fir.heap<!fir.array<?x?xf32>>>>{{.*}},
! CHECK-SAME:  %[[VAL_1:.*]]: !fir.ref<!fir.array<2x3xf32>>{{.*}}) {
subroutine test_from_cst_shape_array(x, y)
  real, allocatable  :: x(:, :)
  real :: y(2, 3)
! CHECK:  %[[VAL_2_0:.*]] = arith.constant 2 : index
! CHECK:  %[[VAL_3_0:.*]] = arith.constant 3 : index
! CHECK:  %[[VAL_2:.*]] = arith.constant 2 : index
! CHECK:  %[[VAL_3:.*]] = arith.constant 3 : index
! CHECK:  %[[VAL_6:.*]] = fir.load %[[VAL_0]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?x?xf32>>>>
! CHECK:  %[[VAL_7:.*]] = fir.box_addr %[[VAL_6]] : (!fir.box<!fir.heap<!fir.array<?x?xf32>>>) -> !fir.heap<!fir.array<?x?xf32>>
! CHECK:  %[[VAL_8:.*]] = fir.convert %[[VAL_7]] : (!fir.heap<!fir.array<?x?xf32>>) -> i64
! CHECK:  %[[VAL_9:.*]] = arith.constant 0 : i64
! CHECK:  %[[VAL_10:.*]] = arith.cmpi ne, %[[VAL_8]], %[[VAL_9]] : i64
! CHECK:  %[[VAL_11:.*]]:2 = fir.if %[[VAL_10]] -> (i1, !fir.heap<!fir.array<?x?xf32>>) {
! CHECK:    %[[VAL_12:.*]] = arith.constant false
! CHECK:    %[[VAL_13:.*]] = arith.constant 0 : index
! CHECK:    %[[VAL_14:.*]]:3 = fir.box_dims %[[VAL_6]], %[[VAL_13]] : (!fir.box<!fir.heap<!fir.array<?x?xf32>>>, index) -> (index, index, index)
! CHECK:    %[[VAL_15:.*]] = arith.constant 1 : index
! CHECK:    %[[VAL_16:.*]]:3 = fir.box_dims %[[VAL_6]], %[[VAL_15]] : (!fir.box<!fir.heap<!fir.array<?x?xf32>>>, index) -> (index, index, index)
! CHECK:    %[[VAL_17:.*]] = arith.cmpi ne, %[[VAL_14]]#1, %[[VAL_2]] : index
! CHECK:    %[[VAL_18:.*]] = arith.select %[[VAL_17]], %[[VAL_17]], %[[VAL_12]] : i1
! CHECK:    %[[VAL_19:.*]] = arith.cmpi ne, %[[VAL_16]]#1, %[[VAL_3]] : index
! CHECK:    %[[VAL_20:.*]] = arith.select %[[VAL_19]], %[[VAL_19]], %[[VAL_18]] : i1
! CHECK:    %[[VAL_21:.*]] = fir.if %[[VAL_20]] -> (!fir.heap<!fir.array<?x?xf32>>) {
! CHECK:      %[[VAL_22:.*]] = fir.allocmem !fir.array<?x?xf32>, %[[VAL_2]], %[[VAL_3]] {uniq_name = ".auto.alloc"}
! CHECK:      fir.result %[[VAL_22]] : !fir.heap<!fir.array<?x?xf32>>
! CHECK:    } else {
! CHECK:      fir.result %[[VAL_7]] : !fir.heap<!fir.array<?x?xf32>>
! CHECK:    }
! CHECK:    fir.result %[[VAL_20]], %[[VAL_23:.*]] : i1, !fir.heap<!fir.array<?x?xf32>>
! CHECK:  } else {
! CHECK:    %[[VAL_24:.*]] = arith.constant true
! CHECK:    %[[VAL_25:.*]] = fir.allocmem !fir.array<?x?xf32>, %[[VAL_2]], %[[VAL_3]] {uniq_name = ".auto.alloc"}
! CHECK:    fir.result %[[VAL_24]], %[[VAL_25]] : i1, !fir.heap<!fir.array<?x?xf32>>
! CHECK:  }

! CHECK:  %[[VAL_26:.*]] = fir.shape %[[VAL_2]], %[[VAL_3]] : (index, index) -> !fir.shape<2>
! CHECK:  %[[VAL_27:.*]] = fir.array_load %[[VAL_11]]#1(%[[VAL_26]]) : (!fir.heap<!fir.array<?x?xf32>>, !fir.shape<2>) -> !fir.array<?x?xf32>
! normal array assignment ....
! CHECK:  fir.array_merge_store %{{.*}}, %{{.*}} to %[[VAL_11]]#1 : !fir.array<?x?xf32>, !fir.array<?x?xf32>, !fir.heap<!fir.array<?x?xf32>>

! CHECK:  fir.if %[[VAL_11]]#0 {
! CHECK:    fir.if %[[VAL_10]] {
! CHECK:      fir.freemem %[[VAL_7]]
! CHECK:    }
! CHECK:    %[[VAL_43:.*]] = fir.shape %[[VAL_2]], %[[VAL_3]] : (index, index) -> !fir.shape<2>
! CHECK:    %[[VAL_44:.*]] = fir.embox %[[VAL_11]]#1(%[[VAL_43]]) : (!fir.heap<!fir.array<?x?xf32>>, !fir.shape<2>) -> !fir.box<!fir.heap<!fir.array<?x?xf32>>>
! CHECK:    fir.store %[[VAL_44]] to %[[VAL_0]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?x?xf32>>>>
! CHECK:  }
  x = y
end subroutine

! CHECK-LABEL: func @_QMalloc_assignPtest_from_dyn_shape_array(
! CHECK-SAME:  %[[VAL_0:.*]]: !fir.ref<!fir.box<!fir.heap<!fir.array<?x?xf32>>>>{{.*}},
! CHECK-SAME:  %[[VAL_1:.*]]: !fir.box<!fir.array<?x?xf32>>{{.*}}) {
subroutine test_from_dyn_shape_array(x, y)
  real, allocatable  :: x(:, :)
  real :: y(:, :)
  x = y
! CHECK:  %[[VAL_3:.*]] = arith.constant 0 : index
! CHECK:  %[[VAL_4:.*]]:3 = fir.box_dims %[[VAL_1]], %[[VAL_3]] : (!fir.box<!fir.array<?x?xf32>>, index) -> (index, index, index)
! CHECK:  %[[VAL_5:.*]] = arith.constant 1 : index
! CHECK:  %[[VAL_6:.*]]:3 = fir.box_dims %[[VAL_1]], %[[VAL_5]] : (!fir.box<!fir.array<?x?xf32>>, index) -> (index, index, index)
! CHECK:  %[[VAL_7:.*]] = fir.load %[[VAL_0]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?x?xf32>>>>
! CHECK:  %[[VAL_8:.*]] = fir.box_addr %[[VAL_7]] : (!fir.box<!fir.heap<!fir.array<?x?xf32>>>) -> !fir.heap<!fir.array<?x?xf32>>
! CHECK:  %[[VAL_9:.*]] = fir.convert %[[VAL_8]] : (!fir.heap<!fir.array<?x?xf32>>) -> i64
! CHECK:  %[[VAL_10:.*]] = arith.constant 0 : i64
! CHECK:  %[[VAL_11:.*]] = arith.cmpi ne, %[[VAL_9]], %[[VAL_10]] : i64
! CHECK:  %[[VAL_12:.*]]:2 = fir.if %[[VAL_11]] -> (i1, !fir.heap<!fir.array<?x?xf32>>) {
! CHECK:    %[[VAL_13:.*]] = arith.constant false
! CHECK:    %[[VAL_14:.*]] = arith.constant 0 : index
! CHECK:    %[[VAL_15:.*]]:3 = fir.box_dims %[[VAL_7]], %[[VAL_14]] : (!fir.box<!fir.heap<!fir.array<?x?xf32>>>, index) -> (index, index, index)
! CHECK:    %[[VAL_16:.*]] = arith.constant 1 : index
! CHECK:    %[[VAL_17:.*]]:3 = fir.box_dims %[[VAL_7]], %[[VAL_16]] : (!fir.box<!fir.heap<!fir.array<?x?xf32>>>, index) -> (index, index, index)
! CHECK:    %[[VAL_18:.*]] = arith.cmpi ne, %[[VAL_15]]#1, %[[VAL_4]]#1 : index
! CHECK:    %[[VAL_19:.*]] = arith.select %[[VAL_18]], %[[VAL_18]], %[[VAL_13]] : i1
! CHECK:    %[[VAL_20:.*]] = arith.cmpi ne, %[[VAL_17]]#1, %[[VAL_6]]#1 : index
! CHECK:    %[[VAL_21:.*]] = arith.select %[[VAL_20]], %[[VAL_20]], %[[VAL_19]] : i1
! CHECK:    %[[VAL_22:.*]] = fir.if %[[VAL_21]] -> (!fir.heap<!fir.array<?x?xf32>>) {
! CHECK:      %[[VAL_23:.*]] = fir.allocmem !fir.array<?x?xf32>, %[[VAL_4]]#1, %[[VAL_6]]#1 {uniq_name = ".auto.alloc"}
! CHECK:      fir.result %[[VAL_23]] : !fir.heap<!fir.array<?x?xf32>>
! CHECK:    } else {
! CHECK:      fir.result %[[VAL_8]] : !fir.heap<!fir.array<?x?xf32>>
! CHECK:    }
! CHECK:    fir.result %[[VAL_21]], %[[VAL_24:.*]] : i1, !fir.heap<!fir.array<?x?xf32>>
! CHECK:  } else {
! CHECK:    %[[VAL_25:.*]] = arith.constant true
! CHECK:    %[[VAL_26:.*]] = fir.allocmem !fir.array<?x?xf32>, %[[VAL_4]]#1, %[[VAL_6]]#1 {uniq_name = ".auto.alloc"}
! CHECK:    fir.result %[[VAL_25]], %[[VAL_26]] : i1, !fir.heap<!fir.array<?x?xf32>>
! CHECK:  }

! CHECK:  %[[VAL_27:.*]] = fir.shape %[[VAL_4]]#1, %[[VAL_6]]#1 : (index, index) -> !fir.shape<2>
! CHECK:  %[[VAL_28:.*]] = fir.array_load %[[VAL_12]]#1(%[[VAL_27]]) : (!fir.heap<!fir.array<?x?xf32>>, !fir.shape<2>) -> !fir.array<?x?xf32>
! normal array assignment ....
! CHECK:  fir.array_merge_store %{{.*}}, %{{.*}} to %[[VAL_12]]#1 : !fir.array<?x?xf32>, !fir.array<?x?xf32>, !fir.heap<!fir.array<?x?xf32>>

! CHECK:  fir.if %[[VAL_12]]#0 {
! CHECK:    fir.if %[[VAL_11]] {
! CHECK:      fir.freemem %[[VAL_8]]
! CHECK:    }
! CHECK:    %[[VAL_44:.*]] = fir.shape %[[VAL_4]]#1, %[[VAL_6]]#1 : (index, index) -> !fir.shape<2>
! CHECK:    %[[VAL_45:.*]] = fir.embox %[[VAL_12]]#1(%[[VAL_44]]) : (!fir.heap<!fir.array<?x?xf32>>, !fir.shape<2>) -> !fir.box<!fir.heap<!fir.array<?x?xf32>>>
! CHECK:    fir.store %[[VAL_45]] to %[[VAL_0]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?x?xf32>>>>
! CHECK:  }
end subroutine

! CHECK-LABEL: func @_QMalloc_assignPtest_with_lbounds(
! CHECK-SAME:  %[[VAL_0:.*]]: !fir.ref<!fir.box<!fir.heap<!fir.array<?x?xf32>>>>{{.*}},
! CHECK-SAME:  %[[VAL_1:.*]]: !fir.box<!fir.array<?x?xf32>>{{.*}}) {
subroutine test_with_lbounds(x, y)
  real, allocatable  :: x(:, :)
  real :: y(10:, 20:)
! CHECK:  %[[VAL_2:.*]] = arith.constant 10 : i64
! CHECK:  %[[VAL_3:.*]] = fir.convert %[[VAL_2]] : (i64) -> index
! CHECK:  %[[VAL_4:.*]] = arith.constant 20 : i64
! CHECK:  %[[VAL_5:.*]] = fir.convert %[[VAL_4]] : (i64) -> index
! CHECK:  %[[VAL_8:.*]] = arith.constant 0 : index
! CHECK:  %[[VAL_9:.*]]:3 = fir.box_dims %[[VAL_1]], %[[VAL_8]] : (!fir.box<!fir.array<?x?xf32>>, index) -> (index, index, index)
! CHECK:  %[[VAL_10:.*]] = arith.constant 1 : index
! CHECK:  %[[VAL_11:.*]]:3 = fir.box_dims %[[VAL_1]], %[[VAL_10]] : (!fir.box<!fir.array<?x?xf32>>, index) -> (index, index, index)
! CHECK:  %[[VAL_12:.*]] = fir.load %[[VAL_0]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?x?xf32>>>>
! CHECK:  %[[VAL_13:.*]] = fir.box_addr %[[VAL_12]] : (!fir.box<!fir.heap<!fir.array<?x?xf32>>>) -> !fir.heap<!fir.array<?x?xf32>>
! CHECK:  %[[VAL_14:.*]] = fir.convert %[[VAL_13]] : (!fir.heap<!fir.array<?x?xf32>>) -> i64
! CHECK:  %[[VAL_15:.*]] = arith.constant 0 : i64
! CHECK:  %[[VAL_16:.*]] = arith.cmpi ne, %[[VAL_14]], %[[VAL_15]] : i64
! CHECK:  %[[VAL_17:.*]]:2 = fir.if %[[VAL_16]] -> (i1, !fir.heap<!fir.array<?x?xf32>>) {
! CHECK:    %[[VAL_18:.*]] = arith.constant false
! CHECK:    %[[VAL_19:.*]] = arith.constant 0 : index
! CHECK:    %[[VAL_20:.*]]:3 = fir.box_dims %[[VAL_12]], %[[VAL_19]] : (!fir.box<!fir.heap<!fir.array<?x?xf32>>>, index) -> (index, index, index)
! CHECK:    %[[VAL_21:.*]] = arith.constant 1 : index
! CHECK:    %[[VAL_22:.*]]:3 = fir.box_dims %[[VAL_12]], %[[VAL_21]] : (!fir.box<!fir.heap<!fir.array<?x?xf32>>>, index) -> (index, index, index)
! CHECK:    %[[VAL_23:.*]] = arith.cmpi ne, %[[VAL_20]]#1, %[[VAL_9]]#1 : index
! CHECK:    %[[VAL_24:.*]] = arith.select %[[VAL_23]], %[[VAL_23]], %[[VAL_18]] : i1
! CHECK:    %[[VAL_25:.*]] = arith.cmpi ne, %[[VAL_22]]#1, %[[VAL_11]]#1 : index
! CHECK:    %[[VAL_26:.*]] = arith.select %[[VAL_25]], %[[VAL_25]], %[[VAL_24]] : i1
! CHECK:    %[[VAL_27:.*]] = fir.if %[[VAL_26]] -> (!fir.heap<!fir.array<?x?xf32>>) {
! CHECK:      %[[VAL_28:.*]] = fir.allocmem !fir.array<?x?xf32>, %[[VAL_9]]#1, %[[VAL_11]]#1 {uniq_name = ".auto.alloc"}
! CHECK:      fir.result %[[VAL_28]] : !fir.heap<!fir.array<?x?xf32>>
! CHECK:    } else {
! CHECK:      fir.result %[[VAL_13]] : !fir.heap<!fir.array<?x?xf32>>
! CHECK:    }
! CHECK:    fir.result %[[VAL_26]], %[[VAL_29:.*]] : i1, !fir.heap<!fir.array<?x?xf32>>
! CHECK:  } else {
! CHECK:    %[[VAL_30:.*]] = arith.constant true
! CHECK:    %[[VAL_31:.*]] = fir.allocmem !fir.array<?x?xf32>, %[[VAL_9]]#1, %[[VAL_11]]#1 {uniq_name = ".auto.alloc"}
! CHECK:    fir.result %[[VAL_30]], %[[VAL_31]] : i1, !fir.heap<!fir.array<?x?xf32>>
! CHECK:  }

! CHECK:  %[[VAL_32:.*]] = fir.shape %[[VAL_9]]#1, %[[VAL_11]]#1 : (index, index) -> !fir.shape<2>
! CHECK:  %[[VAL_33:.*]] = fir.array_load %[[VAL_17]]#1(%[[VAL_32]]) : (!fir.heap<!fir.array<?x?xf32>>, !fir.shape<2>) -> !fir.array<?x?xf32>
! normal array assignment ....
! CHECK:  fir.array_merge_store %{{.*}}, %{{.*}} to %[[VAL_17]]#1 : !fir.array<?x?xf32>, !fir.array<?x?xf32>, !fir.heap<!fir.array<?x?xf32>>

! CHECK:  fir.if %[[VAL_17]]#0 {
! CHECK:    fir.if %[[VAL_16]] {
! CHECK:      fir.freemem %[[VAL_13]]
! CHECK:    }
! CHECK:    %[[VAL_49:.*]] = fir.shape_shift %[[VAL_3]], %[[VAL_9]]#1, %[[VAL_5]], %[[VAL_11]]#1 : (index, index, index, index) -> !fir.shapeshift<2>
! CHECK:    %[[VAL_50:.*]] = fir.embox %[[VAL_17]]#1(%[[VAL_49]]) : (!fir.heap<!fir.array<?x?xf32>>, !fir.shapeshift<2>) -> !fir.box<!fir.heap<!fir.array<?x?xf32>>>
! CHECK:    fir.store %[[VAL_50]] to %[[VAL_0]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?x?xf32>>>>
! CHECK:  }
  x = y
end subroutine

! CHECK-LABEL: func @_QMalloc_assignPtest_runtime_shape(
! CHECK-SAME:  %[[VAL_0:.*]]: !fir.ref<!fir.box<!fir.heap<!fir.array<?x?xf32>>>>{{.*}}) {
subroutine test_runtime_shape(x)
  real, allocatable  :: x(:, :)
  interface
   function return_pointer()
     real, pointer :: return_pointer(:, :)
   end function
  end interface
! CHECK:  %[[VAL_1:.*]] = fir.alloca !fir.box<!fir.ptr<!fir.array<?x?xf32>>> {bindc_name = ".result"}
! CHECK:  %[[VAL_2:.*]] = fir.call @_QPreturn_pointer() : () -> !fir.box<!fir.ptr<!fir.array<?x?xf32>>>
! CHECK:  fir.save_result %[[VAL_2]] to %[[VAL_1]] : !fir.box<!fir.ptr<!fir.array<?x?xf32>>>, !fir.ref<!fir.box<!fir.ptr<!fir.array<?x?xf32>>>>
! CHECK:  %[[VAL_3:.*]] = fir.load %[[VAL_1]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?x?xf32>>>>
! CHECK:  %[[VAL_4:.*]] = arith.constant 0 : index
! CHECK:  %[[VAL_5:.*]]:3 = fir.box_dims %[[VAL_3]], %[[VAL_4]] : (!fir.box<!fir.ptr<!fir.array<?x?xf32>>>, index) -> (index, index, index)
! CHECK:  %[[VAL_6:.*]] = arith.constant 1 : index
! CHECK:  %[[VAL_7:.*]]:3 = fir.box_dims %[[VAL_3]], %[[VAL_6]] : (!fir.box<!fir.ptr<!fir.array<?x?xf32>>>, index) -> (index, index, index)
! CHECK:  %[[VAL_8:.*]] = fir.shift %[[VAL_5]]#0, %[[VAL_7]]#0 : (index, index) -> !fir.shift<2>
! CHECK:  %[[VAL_10:.*]] = arith.constant 0 : index
! CHECK:  %[[VAL_11:.*]]:3 = fir.box_dims %[[VAL_3]], %[[VAL_10]] : (!fir.box<!fir.ptr<!fir.array<?x?xf32>>>, index) -> (index, index, index)
! CHECK:  %[[VAL_12:.*]] = arith.constant 1 : index
! CHECK:  %[[VAL_13:.*]]:3 = fir.box_dims %[[VAL_3]], %[[VAL_12]] : (!fir.box<!fir.ptr<!fir.array<?x?xf32>>>, index) -> (index, index, index)
! CHECK:  %[[VAL_9:.*]] = fir.array_load %[[VAL_3]](%[[VAL_8]]) : (!fir.box<!fir.ptr<!fir.array<?x?xf32>>>, !fir.shift<2>) -> !fir.array<?x?xf32>
! CHECK:  %[[VAL_14:.*]] = fir.load %[[VAL_0]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?x?xf32>>>>
! CHECK:  %[[VAL_15:.*]] = fir.box_addr %[[VAL_14]] : (!fir.box<!fir.heap<!fir.array<?x?xf32>>>) -> !fir.heap<!fir.array<?x?xf32>>
! CHECK:  %[[VAL_16:.*]] = fir.convert %[[VAL_15]] : (!fir.heap<!fir.array<?x?xf32>>) -> i64
! CHECK:  %[[VAL_17:.*]] = arith.constant 0 : i64
! CHECK:  %[[VAL_18:.*]] = arith.cmpi ne, %[[VAL_16]], %[[VAL_17]] : i64
! CHECK:  %[[VAL_19:.*]]:2 = fir.if %[[VAL_18]] -> (i1, !fir.heap<!fir.array<?x?xf32>>) {
! CHECK:    %[[VAL_20:.*]] = arith.constant false
! CHECK:    %[[VAL_21:.*]] = arith.constant 0 : index
! CHECK:    %[[VAL_22:.*]]:3 = fir.box_dims %[[VAL_14]], %[[VAL_21]] : (!fir.box<!fir.heap<!fir.array<?x?xf32>>>, index) -> (index, index, index)
! CHECK:    %[[VAL_23:.*]] = arith.constant 1 : index
! CHECK:    %[[VAL_24:.*]]:3 = fir.box_dims %[[VAL_14]], %[[VAL_23]] : (!fir.box<!fir.heap<!fir.array<?x?xf32>>>, index) -> (index, index, index)
! CHECK:    %[[VAL_25:.*]] = arith.cmpi ne, %[[VAL_22]]#1, %[[VAL_11]]#1 : index
! CHECK:    %[[VAL_26:.*]] = arith.select %[[VAL_25]], %[[VAL_25]], %[[VAL_20]] : i1
! CHECK:    %[[VAL_27:.*]] = arith.cmpi ne, %[[VAL_24]]#1, %[[VAL_13]]#1 : index
! CHECK:    %[[VAL_28:.*]] = arith.select %[[VAL_27]], %[[VAL_27]], %[[VAL_26]] : i1
! CHECK:    %[[VAL_29:.*]] = fir.if %[[VAL_28]] -> (!fir.heap<!fir.array<?x?xf32>>) {
! CHECK:      %[[VAL_30:.*]] = fir.allocmem !fir.array<?x?xf32>, %[[VAL_11]]#1, %[[VAL_13]]#1 {uniq_name = ".auto.alloc"}
! CHECK:      fir.result %[[VAL_30]] : !fir.heap<!fir.array<?x?xf32>>
! CHECK:    } else {
! CHECK:      fir.result %[[VAL_15]] : !fir.heap<!fir.array<?x?xf32>>
! CHECK:    }
! CHECK:    fir.result %[[VAL_28]], %[[VAL_31:.*]] : i1, !fir.heap<!fir.array<?x?xf32>>
! CHECK:  } else {
! CHECK:    %[[VAL_32:.*]] = arith.constant true
! CHECK:    %[[VAL_33:.*]] = fir.allocmem !fir.array<?x?xf32>, %[[VAL_11]]#1, %[[VAL_13]]#1 {uniq_name = ".auto.alloc"}
! CHECK:    fir.result %[[VAL_32]], %[[VAL_33]] : i1, !fir.heap<!fir.array<?x?xf32>>
! CHECK:  }

! CHECK-NOT: fir.call @_QPreturn_pointer()
! CHECK:  %[[VAL_34:.*]] = fir.shape %[[VAL_11]]#1, %[[VAL_13]]#1 : (index, index) -> !fir.shape<2>
! CHECK:  %[[VAL_35:.*]] = fir.array_load %[[VAL_19]]#1(%[[VAL_34]]) : (!fir.heap<!fir.array<?x?xf32>>, !fir.shape<2>) -> !fir.array<?x?xf32>
! normal array assignment ....
! CHECK-NOT: fir.call @_QPreturn_pointer()
! CHECK:  fir.array_merge_store %{{.*}}, %{{.*}} to %[[VAL_19]]#1 : !fir.array<?x?xf32>, !fir.array<?x?xf32>, !fir.heap<!fir.array<?x?xf32>>
! CHECK-NOT: fir.call @_QPreturn_pointer()

! CHECK:  fir.if %[[VAL_19]]#0 {
! CHECK:    fir.if %[[VAL_18]] {
! CHECK:      fir.freemem %[[VAL_15]]
! CHECK:    }
! CHECK:    %[[VAL_56:.*]] = fir.shape %[[VAL_11]]#1, %[[VAL_13]]#1 : (index, index) -> !fir.shape<2>
! CHECK:    %[[VAL_57:.*]] = fir.embox %[[VAL_19]]#1(%[[VAL_56]]) : (!fir.heap<!fir.array<?x?xf32>>, !fir.shape<2>) -> !fir.box<!fir.heap<!fir.array<?x?xf32>>>
! CHECK:    fir.store %[[VAL_57]] to %[[VAL_0]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?x?xf32>>>>
! CHECK:  }
  x = return_pointer()
end subroutine

! CHECK-LABEL: func @_QMalloc_assignPtest_scalar_rhs(
subroutine test_scalar_rhs(x, y)
  real, allocatable  :: x(:)
  real :: y
  ! CHECK: fir.if %{{.*}} -> {{.*}} {
  ! CHECK:   fir.if %false -> {{.*}} {
  ! CHECK:   }
  ! CHECK: } else {
  ! CHECK: %[[error_msg_addr:.*]] = fir.address_of(@[[error_message:.*]]) : !fir.ref<!fir.char<1,76>>
  ! CHECK: %[[msg_addr_cast:.*]] = fir.convert %[[error_msg_addr]] : (!fir.ref<!fir.char<1,76>>) -> !fir.ref<i8>
  ! CHECK: %15 = fir.call @_FortranAReportFatalUserError(%[[msg_addr_cast]], %{{.*}}, %{{.*}}) : (!fir.ref<i8>, !fir.ref<i8>, i32) -> none
  ! CHECK-NOT: allocmem
  ! CHECK: }
  x = y
end subroutine

! -----------------------------------------------------------------------------
!            Test character array RHS
! -----------------------------------------------------------------------------


! Hit TODO: gathering lhs length in array expression
!subroutine test_deferred_char_rhs_scalar(x)
!  character(:), allocatable  :: x(:)
!  x = "Hello world!"
!end subroutine

! CHECK: func @_QMalloc_assignPtest_cst_char_rhs_scalar(
subroutine test_cst_char_rhs_scalar(x)
  character(10), allocatable  :: x(:)
  x = "Hello world!"
  ! CHECK: fir.if %{{.*}} -> {{.*}} {
  ! CHECK:   fir.if %false -> {{.*}} {
  ! CHECK:   }
  ! CHECK: } else {
  ! TODO: runtime error if unallocated
  ! CHECK-NOT: allocmem
  ! CHECK: }
end subroutine

! CHECK: func @_QMalloc_assignPtest_dyn_char_rhs_scalar(
subroutine test_dyn_char_rhs_scalar(x, n)
  integer :: n
  character(n), allocatable  :: x(:)
  x = "Hello world!"
  ! CHECK: fir.if %{{.*}} -> {{.*}} {
  ! CHECK:   fir.if %false -> {{.*}} {
  ! CHECK:   }
  ! CHECK: } else {
  ! TODO: runtime error if unallocated
  ! CHECK-NOT: allocmem
  ! CHECK: }
end subroutine

! Hit TODO: gathering lhs length in array expression
!subroutine test_deferred_char(x, c)
!  character(:), allocatable  :: x(:)
!  character(12) :: c(20)
!  x = "Hello world!"
!end subroutine

! CHECK-LABEL: func @_QMalloc_assignPtest_cst_char(
! CHECK-SAME:  %[[VAL_0:.*]]: !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.char<1,10>>>>>{{.*}},
! CHECK-SAME:  %[[VAL_1:.*]]: !fir.boxchar<1>{{.*}}) {
subroutine test_cst_char(x, c)
  character(10), allocatable  :: x(:)
  character(12) :: c(20)
! CHECK:  %[[VAL_2:.*]]:2 = fir.unboxchar %[[VAL_1]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK:  %[[VAL_3:.*]] = fir.convert %[[VAL_2]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.array<20x!fir.char<1,12>>>
! CHECK:  %[[VAL_4_0:.*]] = arith.constant 20 : index
! CHECK:  %[[VAL_4:.*]] = arith.constant 20 : index
! CHECK:  %[[VAL_5:.*]] = fir.shape %[[VAL_4_0]] : (index) -> !fir.shape<1>
! CHECK:  %[[VAL_7:.*]] = fir.load %[[VAL_0]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.char<1,10>>>>>
! CHECK:  %[[VAL_8:.*]] = fir.box_addr %[[VAL_7]] : (!fir.box<!fir.heap<!fir.array<?x!fir.char<1,10>>>>) -> !fir.heap<!fir.array<?x!fir.char<1,10>>>
! CHECK:  %[[VAL_9:.*]] = fir.convert %[[VAL_8]] : (!fir.heap<!fir.array<?x!fir.char<1,10>>>) -> i64
! CHECK:  %[[VAL_10:.*]] = arith.constant 0 : i64
! CHECK:  %[[VAL_11:.*]] = arith.cmpi ne, %[[VAL_9]], %[[VAL_10]] : i64
! CHECK:  %[[VAL_12:.*]]:2 = fir.if %[[VAL_11]] -> (i1, !fir.heap<!fir.array<?x!fir.char<1,10>>>) {
! CHECK:    %[[VAL_13:.*]] = arith.constant false
! CHECK:    %[[VAL_14:.*]] = arith.constant 0 : index
! CHECK:    %[[VAL_15:.*]]:3 = fir.box_dims %[[VAL_7]], %[[VAL_14]] : (!fir.box<!fir.heap<!fir.array<?x!fir.char<1,10>>>>, index) -> (index, index, index)
! CHECK:    %[[VAL_16:.*]] = arith.cmpi ne, %[[VAL_15]]#1, %[[VAL_4]] : index
! CHECK:    %[[VAL_17:.*]] = arith.select %[[VAL_16]], %[[VAL_16]], %[[VAL_13]] : i1
! CHECK:    %[[VAL_18:.*]] = fir.if %[[VAL_17]] -> (!fir.heap<!fir.array<?x!fir.char<1,10>>>) {
! CHECK:      %[[VAL_19:.*]] = fir.allocmem !fir.array<?x!fir.char<1,10>>, %[[VAL_4]] {uniq_name = ".auto.alloc"}
! CHECK:      fir.result %[[VAL_19]] : !fir.heap<!fir.array<?x!fir.char<1,10>>>
! CHECK:    } else {
! CHECK:      fir.result %[[VAL_8]] : !fir.heap<!fir.array<?x!fir.char<1,10>>>
! CHECK:    }
! CHECK:    fir.result %[[VAL_17]], %[[VAL_20:.*]] : i1, !fir.heap<!fir.array<?x!fir.char<1,10>>>
! CHECK:  } else {
! CHECK:    %[[VAL_21:.*]] = arith.constant true
! CHECK:    %[[VAL_22:.*]] = fir.allocmem !fir.array<?x!fir.char<1,10>>, %[[VAL_4]] {uniq_name = ".auto.alloc"}
! CHECK:    fir.result %[[VAL_21]], %[[VAL_22]] : i1, !fir.heap<!fir.array<?x!fir.char<1,10>>>
! CHECK:  }

! CHECK:  %[[VAL_23:.*]] = fir.shape %[[VAL_4]] : (index) -> !fir.shape<1>
! CHECK:  %[[VAL_24:.*]] = fir.array_load %[[VAL_12]]#1(%[[VAL_23]]) : (!fir.heap<!fir.array<?x!fir.char<1,10>>>, !fir.shape<1>) -> !fir.array<?x!fir.char<1,10>>
! CHECK:  fir.array_merge_store %{{.*}}, %{{.*}} to %[[VAL_12]]#1 : !fir.array<?x!fir.char<1,10>>, !fir.array<?x!fir.char<1,10>>, !fir.heap<!fir.array<?x!fir.char<1,10>>>
! CHECK:  fir.if %[[VAL_12]]#0 {
! CHECK:    fir.if %[[VAL_11]] {
! CHECK:      fir.freemem %[[VAL_8]]
! CHECK:    }
! CHECK:    %[[VAL_36:.*]] = fir.shape %[[VAL_4]] : (index) -> !fir.shape<1>
! CHECK:    %[[VAL_37:.*]] = fir.embox %[[VAL_12]]#1(%[[VAL_36]]) : (!fir.heap<!fir.array<?x!fir.char<1,10>>>, !fir.shape<1>) -> !fir.box<!fir.heap<!fir.array<?x!fir.char<1,10>>>>
! CHECK:    fir.store %[[VAL_37]] to %[[VAL_0]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.char<1,10>>>>>
! CHECK:  }
  x = c
end subroutine

! CHECK-LABEL: func @_QMalloc_assignPtest_dyn_char(
! CHECK-SAME:  %[[VAL_0:.*]]: !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.char<1,?>>>>>{{.*}}, %[[VAL_1:.*]]: !fir.ref<i32>{{.*}}, %[[VAL_2:.*]]: !fir.boxchar<1>{{.*}}) {
subroutine test_dyn_char(x, n, c)
  integer :: n
  character(n), allocatable  :: x(:)
  character(*) :: c(20)
! CHECK:  %[[VAL_3:.*]]:2 = fir.unboxchar %[[VAL_2]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
! CHECK:  %[[VAL_4:.*]] = fir.convert %[[VAL_3]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.array<20x!fir.char<1,?>>>
! CHECK:  %[[VAL_5_0:.*]] = arith.constant 20 : index
! CHECK:  %[[VAL_6A:.*]] = fir.load %[[VAL_1]] : !fir.ref<i32>
! CHECK:  %[[c0_i32:.*]] = arith.constant 0 : i32
! CHECK:  %[[VAL_6B:.*]] = arith.cmpi sgt, %[[VAL_6A]], %[[c0_i32]] : i32
! CHECK:  %[[VAL_6:.*]] = arith.select %[[VAL_6B]], %[[VAL_6A]], %[[c0_i32]] : i32
! CHECK:  %[[VAL_5:.*]] = arith.constant 20 : index
! CHECK:  %[[VAL_7:.*]] = fir.shape %[[VAL_5_0]] : (index) -> !fir.shape<1>
! CHECK:  %[[VAL_9:.*]] = fir.load %[[VAL_0]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.char<1,?>>>>>
! CHECK:  %[[VAL_10:.*]] = fir.box_addr %[[VAL_9]] : (!fir.box<!fir.heap<!fir.array<?x!fir.char<1,?>>>>) -> !fir.heap<!fir.array<?x!fir.char<1,?>>>
! CHECK:  %[[VAL_11:.*]] = fir.convert %[[VAL_10]] : (!fir.heap<!fir.array<?x!fir.char<1,?>>>) -> i64
! CHECK:  %[[VAL_12:.*]] = arith.constant 0 : i64
! CHECK:  %[[VAL_13:.*]] = arith.cmpi ne, %[[VAL_11]], %[[VAL_12]] : i64
! CHECK:  %[[VAL_14:.*]]:2 = fir.if %[[VAL_13]] -> (i1, !fir.heap<!fir.array<?x!fir.char<1,?>>>) {
! CHECK:    %[[VAL_15:.*]] = arith.constant false
! CHECK:    %[[VAL_16:.*]] = arith.constant 0 : index
! CHECK:    %[[VAL_17:.*]]:3 = fir.box_dims %[[VAL_9]], %[[VAL_16]] : (!fir.box<!fir.heap<!fir.array<?x!fir.char<1,?>>>>, index) -> (index, index, index)
! CHECK:    %[[VAL_18:.*]] = arith.cmpi ne, %[[VAL_17]]#1, %[[VAL_5]] : index
! CHECK:    %[[VAL_19:.*]] = arith.select %[[VAL_18]], %[[VAL_18]], %[[VAL_15]] : i1
! CHECK:    %[[VAL_20:.*]] = fir.if %[[VAL_19]] -> (!fir.heap<!fir.array<?x!fir.char<1,?>>>) {
! CHECK:      %[[VAL_21:.*]] = fir.convert %[[VAL_6]] : (i32) -> index
! CHECK:      %[[VAL_22:.*]] = fir.allocmem !fir.array<?x!fir.char<1,?>>(%[[VAL_21]] : index), %[[VAL_5]] {uniq_name = ".auto.alloc"}
! CHECK:      fir.result %[[VAL_22]] : !fir.heap<!fir.array<?x!fir.char<1,?>>>
! CHECK:    } else {
! CHECK:      fir.result %[[VAL_10]] : !fir.heap<!fir.array<?x!fir.char<1,?>>>
! CHECK:    }
! CHECK:    fir.result %[[VAL_19]], %[[VAL_23:.*]] : i1, !fir.heap<!fir.array<?x!fir.char<1,?>>>
! CHECK:  } else {
! CHECK:    %[[VAL_24:.*]] = arith.constant true
! CHECK:    %[[VAL_25:.*]] = fir.convert %[[VAL_6]] : (i32) -> index
! CHECK:    %[[VAL_26:.*]] = fir.allocmem !fir.array<?x!fir.char<1,?>>(%[[VAL_25]] : index), %[[VAL_5]] {uniq_name = ".auto.alloc"}
! CHECK:    fir.result %[[VAL_24]], %[[VAL_26]] : i1, !fir.heap<!fir.array<?x!fir.char<1,?>>>
! CHECK:  }

! CHECK:  %[[VAL_27:.*]] = fir.shape %[[VAL_5]] : (index) -> !fir.shape<1>
! CHECK:  %[[VAL_28:.*]] = fir.array_load %[[VAL_14]]#1(%[[VAL_27]]) typeparams %[[VAL_6]] : (!fir.heap<!fir.array<?x!fir.char<1,?>>>, !fir.shape<1>, i32) -> !fir.array<?x!fir.char<1,?>>
! normal array assignment ....
! CHECK:  fir.array_merge_store %{{.*}}, %{{.*}} to %[[VAL_14]]#1 typeparams %[[VAL_6]] : !fir.array<?x!fir.char<1,?>>, !fir.array<?x!fir.char<1,?>>, !fir.heap<!fir.array<?x!fir.char<1,?>>>, i32

! CHECK:  fir.if %[[VAL_14]]#0 {
! CHECK:    %[[VAL_39:.*]] = fir.convert %[[VAL_6]] : (i32) -> index
! CHECK:    fir.if %[[VAL_13]] {
! CHECK:      fir.freemem %[[VAL_10]]
! CHECK:    }
! CHECK:    %[[VAL_40:.*]] = fir.shape %[[VAL_5]] : (index) -> !fir.shape<1>
! CHECK:    %[[VAL_41:.*]] = fir.embox %[[VAL_14]]#1(%[[VAL_40]]) typeparams %[[VAL_39]] : (!fir.heap<!fir.array<?x!fir.char<1,?>>>, !fir.shape<1>, index) -> !fir.box<!fir.heap<!fir.array<?x!fir.char<1,?>>>>
! CHECK:    fir.store %[[VAL_41]] to %[[VAL_0]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?x!fir.char<1,?>>>>>
! CHECK:  }
  x = c
end subroutine

! CHECK-LABEL: func @_QMalloc_assignPtest_derived_with_init
subroutine test_derived_with_init(x, y)
  type t 
    integer, allocatable :: a(:)
  end type                                                                                     
  type(t), allocatable :: x                                                                    
  type(t) :: y                                                                                 
  ! The allocatable component of `x` need to be initialized
  ! during the automatic allocation (setting its rank and allocation
  ! status) before it is assigned with the component of `y` 
  x = y
! CHECK:  fir.if %{{.*}} {
! CHECK:    %[[VAL_11:.*]] = fir.allocmem !fir.type<_QMalloc_assignFtest_derived_with_initTt{a:!fir.box<!fir.heap<!fir.array<?xi32>>>}> {uniq_name = ".auto.alloc"}
! CHECK:    %[[VAL_12:.*]] = fir.embox %[[VAL_11]] : (!fir.heap<!fir.type<_QMalloc_assignFtest_derived_with_initTt{a:!fir.box<!fir.heap<!fir.array<?xi32>>>}>>) -> !fir.box<!fir.heap<!fir.type<_QMalloc_assignFtest_derived_with_initTt{a:!fir.box<!fir.heap<!fir.array<?xi32>>>}>>>
! CHECK:    %[[VAL_15:.*]] = fir.convert %[[VAL_12]] : (!fir.box<!fir.heap<!fir.type<_QMalloc_assignFtest_derived_with_initTt{a:!fir.box<!fir.heap<!fir.array<?xi32>>>}>>>) -> !fir.box<none>
! CHECK:    fir.call @_FortranAInitialize(%[[VAL_15]], %{{.*}}, %{{.*}}) : (!fir.box<none>, !fir.ref<i8>, i32) -> none
! CHECK:    fir.result %[[VAL_11]] : !fir.heap<!fir.type<_QMalloc_assignFtest_derived_with_initTt{a:!fir.box<!fir.heap<!fir.array<?xi32>>>}>>
! CHECK:  } else {
! CHECK:    fir.result %{{.*}} : !fir.heap<!fir.type<_QMalloc_assignFtest_derived_with_initTt{a:!fir.box<!fir.heap<!fir.array<?xi32>>>}>>
! CHECK:  }
end subroutine

! CHECK-LABEL: func @_QMalloc_assignPtest_vector_subscript(
! CHECK-SAME: %[[VAL_0:.*]]: !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>> {fir.bindc_name = "x"},
! CHECK-SAME: %[[VAL_1:.*]]: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "y"},
! CHECK-SAME: %[[VAL_2:.*]]: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "v"}) {
subroutine test_vector_subscript(x, y, v)
  ! Test that the new shape is computed correctly in presence of
  ! vector subscripts on the RHS and that it is used to allocate
  ! the new storage and to drive the implicit loop.
  integer, allocatable :: x(:)
  integer :: y(:), v(:)
  x = y(v)
! CHECK:         %[[VAL_3:.*]] = arith.constant 1 : index
! CHECK:         %[[VAL_4:.*]] = arith.constant 0 : index
! CHECK:         %[[VAL_5:.*]]:3 = fir.box_dims %[[VAL_1]], %[[VAL_4]] : (!fir.box<!fir.array<?xi32>>, index) -> (index, index, index)
! CHECK:         %[[VAL_6:.*]] = arith.constant 0 : index
! CHECK:         %[[VAL_7:.*]]:3 = fir.box_dims %[[VAL_2]], %[[VAL_6]] : (!fir.box<!fir.array<?xi32>>, index) -> (index, index, index)
! CHECK:         %[[VAL_8:.*]] = fir.array_load %[[VAL_2]] : (!fir.box<!fir.array<?xi32>>) -> !fir.array<?xi32>
! CHECK:         %[[VAL_9:.*]] = arith.cmpi sgt, %[[VAL_7]]#1, %[[VAL_5]]#1 : index
! CHECK:         %[[VAL_10:.*]] = arith.select %[[VAL_9]], %[[VAL_5]]#1, %[[VAL_7]]#1 : index
! CHECK:         fir.if {{.*}} {
! CHECK:           %[[VAL_18:.*]] = arith.constant false
! CHECK:           %[[VAL_20:.*]]:3 = fir.box_dims %{{.*}}, %{{.*}} : (!fir.box<!fir.heap<!fir.array<?xi32>>>, index) -> (index, index, index)
! CHECK:           %[[VAL_21:.*]] = arith.cmpi ne, %[[VAL_20]]#1, %[[VAL_10]] : index
! CHECK:           %[[VAL_22:.*]] = arith.select %[[VAL_21]], %[[VAL_21]], %[[VAL_18]] : i1
! CHECK:           fir.if %[[VAL_22]] {{.*}} {
! CHECK:             %[[VAL_24:.*]] = fir.allocmem !fir.array<?xi32>, %[[VAL_10]] {uniq_name = ".auto.alloc"}
! CHECK:             fir.result %[[VAL_24]] : !fir.heap<!fir.array<?xi32>>
! CHECK:           } else {
! CHECK:             fir.result %{{.*}} : !fir.heap<!fir.array<?xi32>>
! CHECK:           }
! CHECK:           fir.result %{{.*}}, %{{.*}}
! CHECK:         } else {
! CHECK:           %[[VAL_27:.*]] = fir.allocmem !fir.array<?xi32>, %[[VAL_10]] {uniq_name = ".auto.alloc"}
! CHECK:           fir.result %{{.*}}, %[[VAL_27]] : i1, !fir.heap<!fir.array<?xi32>>
! CHECK:         }
! CHECK:         %[[VAL_28:.*]] = fir.shape %[[VAL_10]] : (index) -> !fir.shape<1>
! CHECK:         %[[VAL_29:.*]] = fir.array_load %[[VAL_30:.*]]#1(%[[VAL_28]]) : (!fir.heap<!fir.array<?xi32>>, !fir.shape<1>) -> !fir.array<?xi32>
! CHECK:         %[[VAL_31:.*]] = arith.constant 1 : index
! CHECK:         %[[VAL_32:.*]] = arith.constant 0 : index
! CHECK:         %[[VAL_33:.*]] = arith.subi %[[VAL_10]], %[[VAL_31]] : index
! CHECK:         %[[VAL_34:.*]] = fir.do_loop %[[VAL_35:.*]] = %[[VAL_32]] to %[[VAL_33]] step %[[VAL_31]] {{.*}} {
! CHECK:         }
end subroutine

! CHECK: fir.global linkonce @[[error_message]] constant : !fir.char<1,76> {
! CHECK:   %[[msg:.*]] = fir.string_lit "array left hand side must be allocated when the right hand side is a scalar\00"(76) : !fir.char<1,76>
! CHECK:   fir.has_value %[[msg:.*]] : !fir.char<1,76>
! CHECK: }

end module

!  use alloc_assign
!  real :: y(2, 3) = reshape([1,2,3,4,5,6], [2,3])
!  real, allocatable :: x (:, :)
!  allocate(x(2,2))
!  call test_with_lbounds(x, y) 
!  print *, x(10, 20)
!  print *, x
!end
