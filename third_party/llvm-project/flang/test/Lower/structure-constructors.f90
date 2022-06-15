! Test lowering of structure constructors
! RUN: bbc -emit-fir %s -o - | FileCheck %s

module m_struct_ctor
  implicit none
  type t_simple
    real :: x
  end type
  type t_char_scalar
    real :: x
    character(3) :: c
  end type
  type t_array
    real :: x
    integer :: i(5)
  end type
  type t_char_array
    real :: x
    character(3) :: c(5)
  end type
  type t_ptr
    real :: x
    integer, pointer :: p(:,:)
  end type
  type t_nested
    real :: x
    type(t_array) :: dt
  end type
contains
  ! CHECK-LABEL: func @_QMm_struct_ctorPtest_simple(
  ! CHECK-SAME: %[[x:.*]]: !fir.ref<f32>{{.*}})
  subroutine test_simple(x)
    real :: x
    ! CHECK: %[[tmp:.*]] = fir.alloca !fir.type<_QMm_struct_ctorTt_simple{x:f32}>
    ! CHECK: %[[field:.*]] = fir.field_index x, !fir.type<_QMm_struct_ctorTt_simple{x:f32}>
    ! CHECK: %[[xcoor:.*]] = fir.coordinate_of %[[tmp]], %[[field]] : (!fir.ref<!fir.type<_QMm_struct_ctorTt_simple{x:f32}>>, !fir.field) -> !fir.ref<f32>
    ! CHECK: %[[val:.*]] = fir.load %[[x]] : !fir.ref<f32>
    ! CHECK: fir.store %[[val]] to %[[xcoor]] : !fir.ref<f32>
    call print_simple(t_simple(x=x))
  end subroutine

  ! CHECK-LABEL: func @_QMm_struct_ctorPtest_char_scalar(
  ! CHECK-SAME: %[[x:.*]]: !fir.ref<f32>{{.*}})
  subroutine test_char_scalar(x)
    ! CHECK: %[[tmp:.*]] = fir.alloca !fir.type<_QMm_struct_ctorTt_char_scalar{x:f32,c:!fir.char<1,3>}>
    ! CHECK: %[[xfield:.*]] = fir.field_index x, !fir.type<_QMm_struct_ctorTt_char_scalar{x:f32,c:!fir.char<1,3>}>
    ! CHECK: %[[xcoor:.*]] = fir.coordinate_of %[[tmp]], %[[xfield]] : (!fir.ref<!fir.type<_QMm_struct_ctorTt_char_scalar{x:f32,c:!fir.char<1,3>}>>, !fir.field) -> !fir.ref<f32>
    ! CHECK: %[[val:.*]] = fir.load %[[x]] : !fir.ref<f32>
    ! CHECK: fir.store %[[val]] to %[[xcoor]] : !fir.ref<f32>

    ! CHECK: %[[cfield:.*]] = fir.field_index c, !fir.type<_QMm_struct_ctorTt_char_scalar{x:f32,c:!fir.char<1,3>}>
    ! CHECK: %[[ccoor:.*]] = fir.coordinate_of %[[tmp]], %[[cfield]] : (!fir.ref<!fir.type<_QMm_struct_ctorTt_char_scalar{x:f32,c:!fir.char<1,3>}>>, !fir.field) -> !fir.ref<!fir.char<1,3>>
    ! CHECK: %[[cst:.*]] = fir.address_of(@_QQ{{.*}}) : !fir.ref<!fir.char<1,3>>
    ! CHECK-DAG: %[[ccast:.*]] = fir.convert %[[ccoor]] : (!fir.ref<!fir.char<1,3>>) -> !fir.ref<i8>
    ! CHECK-DAG: %[[cstcast:.*]] = fir.convert %[[cst]] : (!fir.ref<!fir.char<1,3>>) -> !fir.ref<i8>
    ! CHECK: fir.call @llvm.memmove.p0.p0.i64(%[[ccast]], %[[cstcast]], %{{.*}}, %{{.*}}) : (!fir.ref<i8>, !fir.ref<i8>, i64, i1) -> ()
    real :: x
    call print_char_scalar(t_char_scalar(x=x, c="abc"))
  end subroutine

  ! CHECK-LABEL: func @_QMm_struct_ctorPtest_simple_array(
  ! CHECK-SAME: %[[x:.*]]: !fir.ref<f32>{{.*}}, %[[j:.*]]: !fir.ref<!fir.array<5xi32>>{{.*}})
  subroutine test_simple_array(x, j)
    real :: x
    integer :: j(5)
    call print_simple_array(t_array(x=x, i=2*j))
    ! CHECK: %[[tmp:.*]] = fir.alloca !fir.type<_QMm_struct_ctorTt_array{x:f32,i:!fir.array<5xi32>}>
    ! CHECK: %[[xfield:.*]] = fir.field_index x, !fir.type<_QMm_struct_ctorTt_array{x:f32,i:!fir.array<5xi32>}>
    ! CHECK: %[[xcoor:.*]] = fir.coordinate_of %[[tmp]], %[[xfield]] : (!fir.ref<!fir.type<_QMm_struct_ctorTt_array{x:f32,i:!fir.array<5xi32>}>>, !fir.field) -> !fir.ref<f32>
    ! CHECK: %[[val:.*]] = fir.load %[[x]] : !fir.ref<f32>
    ! CHECK: fir.store %[[val]] to %[[xcoor]] : !fir.ref<f32>

    ! CHECK: %[[ifield:.*]] = fir.field_index i, !fir.type<_QMm_struct_ctorTt_array{x:f32,i:!fir.array<5xi32>}>
    ! CHECK: %[[icoor:.*]] = fir.coordinate_of %[[tmp]], %[[ifield]] : (!fir.ref<!fir.type<_QMm_struct_ctorTt_array{x:f32,i:!fir.array<5xi32>}>>, !fir.field) -> !fir.ref<!fir.array<5xi32>>
    ! CHECK: %[[iload:.*]] = fir.array_load %[[icoor]](%{{.*}}) : (!fir.ref<!fir.array<5xi32>>, !fir.shape<1>) -> !fir.array<5xi32>
    ! CHECK: %[[jload:.*]] = fir.array_load %[[j]](%{{.*}}) : (!fir.ref<!fir.array<5xi32>>, !fir.shape<1>) -> !fir.array<5xi32>
    ! CHECK: %[[loop:.*]] = fir.do_loop %[[idx:.*]] = %c0{{.*}} to %{{.*}} step %c1{{.*}} iter_args(%[[res:.*]] = %[[iload]]) -> (!fir.array<5xi32>) {
    ! CHECK:   %[[jval:.*]] = fir.array_fetch %[[jload]], %[[idx]] : (!fir.array<5xi32>, index) -> i32
    ! CHECK:   %[[ival:.*]] = arith.muli %c2{{.*}}, %[[jval]] : i32
    ! CHECK:   %[[iupdate:.*]] = fir.array_update %[[res]], %[[ival]], %[[idx]] : (!fir.array<5xi32>, i32, index) -> !fir.array<5xi32>
    ! CHECK:   fir.result %[[iupdate]] : !fir.array<5xi32>
    ! CHECK: fir.array_merge_store %[[iload]], %[[loop]] to %[[icoor]] : !fir.array<5xi32>, !fir.array<5xi32>, !fir.ref<!fir.array<5xi32>>

  end subroutine

! CHECK-LABEL: func @_QMm_struct_ctorPtest_char_array(
! CHECK-SAME:  %[[VAL_0:.*]]: !fir.ref<f32>{{.*}}, %[[VAL_1:.*]]: !fir.boxchar<1>{{.*}}) {
  subroutine test_char_array(x, c1)
  ! CHECK: %[[VAL_3:.*]] = fir.alloca !fir.type<_QMm_struct_ctorTt_char_array{x:f32,c:!fir.array<5x!fir.char<1,3>>}>
  ! CHECK: %[[VAL_4:.*]]:2 = fir.unboxchar %[[VAL_1]] : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
  ! CHECK: %[[VAL_5:.*]] = fir.convert %[[VAL_4]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.array<5x!fir.char<1,3>>>
  ! CHECK: %[[VAL_6:.*]] = arith.constant 5 : index
  ! CHECK: %[[VAL_7:.*]] = fir.field_index x, !fir.type<_QMm_struct_ctorTt_char_array{x:f32,c:!fir.array<5x!fir.char<1,3>>}>
  ! CHECK: %[[VAL_8:.*]] = fir.coordinate_of %[[VAL_3]], %[[VAL_7]] : (!fir.ref<!fir.type<_QMm_struct_ctorTt_char_array{x:f32,c:!fir.array<5x!fir.char<1,3>>}>>, !fir.field) -> !fir.ref<f32>
  ! CHECK: %[[VAL_9:.*]] = fir.load %[[VAL_0]] : !fir.ref<f32>
  ! CHECK: fir.store %[[VAL_9]] to %[[VAL_8]] : !fir.ref<f32>
  ! CHECK: %[[VAL_10:.*]] = fir.field_index c, !fir.type<_QMm_struct_ctorTt_char_array{x:f32,c:!fir.array<5x!fir.char<1,3>>}>
  ! CHECK: %[[VAL_11:.*]] = fir.coordinate_of %[[VAL_3]], %[[VAL_10]] : (!fir.ref<!fir.type<_QMm_struct_ctorTt_char_array{x:f32,c:!fir.array<5x!fir.char<1,3>>}>>, !fir.field) -> !fir.ref<!fir.array<5x!fir.char<1,3>>>
  ! CHECK: %[[VAL_12:.*]] = arith.constant 5 : index
  ! CHECK: %[[VAL_13:.*]] = fir.shape %[[VAL_12]] : (index) -> !fir.shape<1>
  ! CHECK: %[[VAL_14:.*]] = fir.array_load %[[VAL_11]](%[[VAL_13]]) : (!fir.ref<!fir.array<5x!fir.char<1,3>>>, !fir.shape<1>) -> !fir.array<5x!fir.char<1,3>>
  ! CHECK: %[[VAL_15:.*]] = fir.shape %[[VAL_6]] : (index) -> !fir.shape<1>
  ! CHECK: %[[VAL_16:.*]] = fir.array_load %[[VAL_5]](%[[VAL_15]]) : (!fir.ref<!fir.array<5x!fir.char<1,3>>>, !fir.shape<1>) -> !fir.array<5x!fir.char<1,3>>
  ! CHECK: %[[VAL_17:.*]] = arith.constant 1 : index
  ! CHECK: %[[VAL_18:.*]] = arith.constant 0 : index
  ! CHECK: %[[VAL_19:.*]] = arith.subi %[[VAL_12]], %[[VAL_17]] : index
  ! CHECK: %[[VAL_20:.*]] = fir.do_loop %[[VAL_21:.*]] = %[[VAL_18]] to %[[VAL_19]] step %[[VAL_17]] unordered iter_args(%[[VAL_22:.*]] = %[[VAL_14]]) -> (!fir.array<5x!fir.char<1,3>>) {
  ! CHECK: %[[VAL_23:.*]] = fir.array_access %[[VAL_16]], %[[VAL_21]] : (!fir.array<5x!fir.char<1,3>>, index) -> !fir.ref<!fir.char<1,3>>
  ! CHECK: %[[VAL_24:.*]] = fir.array_access %[[VAL_22]], %[[VAL_21]] : (!fir.array<5x!fir.char<1,3>>, index) -> !fir.ref<!fir.char<1,3>>
  ! CHECK: %[[VAL_25:.*]] = arith.constant 3 : index
  ! CHECK: %[[VAL_26:.*]] = arith.constant 1 : i64
  ! CHECK: %[[VAL_27:.*]] = fir.convert %[[VAL_25]] : (index) -> i64
  ! CHECK: %[[VAL_28:.*]] = arith.muli %[[VAL_26]], %[[VAL_27]] : i64
  ! CHECK: %[[VAL_29:.*]] = arith.constant false
  ! CHECK: %[[VAL_30:.*]] = fir.convert %[[VAL_24]] : (!fir.ref<!fir.char<1,3>>) -> !fir.ref<i8>
  ! CHECK: %[[VAL_31:.*]] = fir.convert %[[VAL_23]] : (!fir.ref<!fir.char<1,3>>) -> !fir.ref<i8>
  ! CHECK: fir.call @llvm.memmove.p0.p0.i64(%[[VAL_30]], %[[VAL_31]], %[[VAL_28]], %[[VAL_29]]) : (!fir.ref<i8>, !fir.ref<i8>, i64, i1) -> ()
  ! CHECK: %[[VAL_32:.*]] = fir.array_amend %[[VAL_22]], %[[VAL_24]] : (!fir.array<5x!fir.char<1,3>>, !fir.ref<!fir.char<1,3>>) -> !fir.array<5x!fir.char<1,3>>
  ! CHECK: fir.result %[[VAL_32]] : !fir.array<5x!fir.char<1,3>>
  ! CHECK: }
  ! CHECK: fir.array_merge_store %[[VAL_14]], %[[VAL_33:.*]] to %[[VAL_11]] : !fir.array<5x!fir.char<1,3>>, !fir.array<5x!fir.char<1,3>>, !fir.ref<!fir.array<5x!fir.char<1,3>>>
  ! CHECK: fir.call @_QMm_struct_ctorPprint_char_array(%[[VAL_3]]) : (!fir.ref<!fir.type<_QMm_struct_ctorTt_char_array{x:f32,c:!fir.array<5x!fir.char<1,3>>}>>) -> ()

    real :: x
    character(3) :: c1(5)
    call print_char_array(t_char_array(x=x, c=c1))
    ! CHECK: return
    ! CHECK: }
  end subroutine

  ! CHECK-LABEL: func @_QMm_struct_ctorPtest_ptr(
  ! CHECK-SAME:    %[[VAL_0:.*]]: !fir.ref<f32>{{.*}}, %[[VAL_1:.*]]: !fir.box<!fir.array<?x?xi32>> {{{.*}}, fir.target}) {
  ! CHECK:         %[[VAL_3:.*]] = fir.alloca !fir.type<_QMm_struct_ctorTt_ptr{x:f32,p:!fir.box<!fir.ptr<!fir.array<?x?xi32>>>}>
  ! CHECK:         %[[VAL_4:.*]] = fir.field_index x, !fir.type<_QMm_struct_ctorTt_ptr{x:f32,p:!fir.box<!fir.ptr<!fir.array<?x?xi32>>>}>
  ! CHECK:         %[[VAL_5:.*]] = fir.coordinate_of %[[VAL_3]], %[[VAL_4]] : (!fir.ref<!fir.type<_QMm_struct_ctorTt_ptr{x:f32,p:!fir.box<!fir.ptr<!fir.array<?x?xi32>>>}>>, !fir.field) -> !fir.ref<f32>
  ! CHECK:         %[[VAL_6:.*]] = fir.load %[[VAL_0]] : !fir.ref<f32>
  ! CHECK:         fir.store %[[VAL_6]] to %[[VAL_5]] : !fir.ref<f32>
  ! CHECK:         %[[VAL_7:.*]] = fir.field_index p, !fir.type<_QMm_struct_ctorTt_ptr{x:f32,p:!fir.box<!fir.ptr<!fir.array<?x?xi32>>>}>
  ! CHECK:         %[[VAL_8:.*]] = fir.coordinate_of %[[VAL_3]], %[[VAL_7]] : (!fir.ref<!fir.type<_QMm_struct_ctorTt_ptr{x:f32,p:!fir.box<!fir.ptr<!fir.array<?x?xi32>>>}>>, !fir.field) -> !fir.ref<!fir.box<!fir.ptr<!fir.array<?x?xi32>>>>
  ! CHECK:         %[[VAL_9:.*]] = arith.constant 1 : i64
  ! CHECK:         %[[VAL_10:.*]] = fir.convert %[[VAL_9]] : (i64) -> index
  ! CHECK:         %[[VAL_11:.*]] = arith.constant 2 : i64
  ! CHECK:         %[[VAL_12:.*]] = fir.convert %[[VAL_11]] : (i64) -> index
  ! CHECK:         %[[VAL_13:.*]] = arith.constant 4 : i64
  ! CHECK:         %[[VAL_14:.*]] = fir.convert %[[VAL_13]] : (i64) -> index
  ! CHECK:         %[[VAL_15:.*]] = arith.constant 1 : i64
  ! CHECK:         %[[VAL_16:.*]] = fir.convert %[[VAL_15]] : (i64) -> index
  ! CHECK:         %[[VAL_17:.*]] = arith.constant 1 : i64
  ! CHECK:         %[[VAL_18:.*]] = fir.convert %[[VAL_17]] : (i64) -> index
  ! CHECK:         %[[VAL_19:.*]] = arith.constant 3 : i64
  ! CHECK:         %[[VAL_20:.*]] = fir.convert %[[VAL_19]] : (i64) -> index
  ! CHECK:         %[[VAL_21:.*]] = fir.slice %[[VAL_10]], %[[VAL_14]], %[[VAL_12]], %[[VAL_16]], %[[VAL_20]], %[[VAL_18]] : (index, index, index, index, index, index) -> !fir.slice<2>
  ! CHECK:         %[[VAL_22:.*]] = fir.rebox %[[VAL_1]] {{\[}}%[[VAL_21]]] : (!fir.box<!fir.array<?x?xi32>>, !fir.slice<2>) -> !fir.box<!fir.array<?x?xi32>>
  ! CHECK:         %[[VAL_23:.*]] = fir.rebox %[[VAL_22]] : (!fir.box<!fir.array<?x?xi32>>) -> !fir.box<!fir.ptr<!fir.array<?x?xi32>>>
  ! CHECK:         fir.store %[[VAL_23]] to %[[VAL_8]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?x?xi32>>>>
  ! CHECK:         fir.call @_QMm_struct_ctorPprint_ptr(%[[VAL_3]]) : (!fir.ref<!fir.type<_QMm_struct_ctorTt_ptr{x:f32,p:!fir.box<!fir.ptr<!fir.array<?x?xi32>>>}>>) -> ()
  ! CHECK:         return
  ! CHECK:       }

  subroutine test_ptr(x, a)
    real :: x
    integer, target :: a(:, :)
    call print_ptr(t_ptr(x=x, p=a(1:4:2, 1:3:1)))
  end subroutine

  ! CHECK-LABEL: func @_QMm_struct_ctorPtest_nested(
  ! CHECK-SAME: %[[VAL_0:.*]]: !fir.ref<f32>{{.*}}, %[[VAL_1:.*]]: !fir.ref<!fir.type<_QMm_struct_ctorTt_array{x:f32,i:!fir.array<5xi32>}>>
  subroutine test_nested(x, d)
    real :: x
    type(t_array) :: d
  ! CHECK:         %[[VAL_2:.*]] = fir.alloca !fir.type<_QMm_struct_ctorTt_nested{x:f32,dt:!fir.type<_QMm_struct_ctorTt_array{x:f32,i:!fir.array<5xi32>}>}>
  ! CHECK:         %[[VAL_3:.*]] = fir.field_index x, !fir.type<_QMm_struct_ctorTt_nested{x:f32,dt:!fir.type<_QMm_struct_ctorTt_array{x:f32,i:!fir.array<5xi32>}>}>
  ! CHECK:         %[[VAL_4:.*]] = fir.coordinate_of %[[VAL_2]], %[[VAL_3]] : (!fir.ref<!fir.type<_QMm_struct_ctorTt_nested{x:f32,dt:!fir.type<_QMm_struct_ctorTt_array{x:f32,i:!fir.array<5xi32>}>}>>, !fir.field) -> !fir.ref<f32>
  ! CHECK:         %[[VAL_5:.*]] = fir.load %[[VAL_0]] : !fir.ref<f32>
  ! CHECK:         fir.store %[[VAL_5]] to %[[VAL_4]] : !fir.ref<f32>
  ! CHECK:         %[[VAL_6:.*]] = fir.field_index dt, !fir.type<_QMm_struct_ctorTt_nested{x:f32,dt:!fir.type<_QMm_struct_ctorTt_array{x:f32,i:!fir.array<5xi32>}>}>
  ! CHECK:         %[[VAL_7:.*]] = fir.coordinate_of %[[VAL_2]], %[[VAL_6]] : (!fir.ref<!fir.type<_QMm_struct_ctorTt_nested{x:f32,dt:!fir.type<_QMm_struct_ctorTt_array{x:f32,i:!fir.array<5xi32>}>}>>, !fir.field) -> !fir.ref<!fir.type<_QMm_struct_ctorTt_array{x:f32,i:!fir.array<5xi32>}>>
  ! CHECK:         %[[VAL_8:.*]] = fir.field_index x, !fir.type<_QMm_struct_ctorTt_array{x:f32,i:!fir.array<5xi32>}>
  ! CHECK:         %[[VAL_9:.*]] = fir.coordinate_of %[[VAL_1]], %[[VAL_8]] : (!fir.ref<!fir.type<_QMm_struct_ctorTt_array{x:f32,i:!fir.array<5xi32>}>>, !fir.field) -> !fir.ref<f32>
  ! CHECK:         %[[VAL_8b:.*]] = fir.field_index x, !fir.type<_QMm_struct_ctorTt_array{x:f32,i:!fir.array<5xi32>}>
  ! CHECK:         %[[VAL_10:.*]] = fir.coordinate_of %[[VAL_7]], %[[VAL_8b]] : (!fir.ref<!fir.type<_QMm_struct_ctorTt_array{x:f32,i:!fir.array<5xi32>}>>, !fir.field) -> !fir.ref<f32>
  ! CHECK:         %[[VAL_11:.*]] = fir.load %[[VAL_9]] : !fir.ref<f32>
  ! CHECK:         fir.store %[[VAL_11]] to %[[VAL_10]] : !fir.ref<f32>
  ! CHECK:         %[[VAL_12:.*]] = fir.field_index i, !fir.type<_QMm_struct_ctorTt_array{x:f32,i:!fir.array<5xi32>}>
  ! CHECK:         %[[VAL_13:.*]] = fir.coordinate_of %[[VAL_1]], %[[VAL_12]] : (!fir.ref<!fir.type<_QMm_struct_ctorTt_array{x:f32,i:!fir.array<5xi32>}>>, !fir.field) -> !fir.ref<!fir.array<5xi32>>
  ! CHECK:         %[[VAL_12b:.*]] = fir.field_index i, !fir.type<_QMm_struct_ctorTt_array{x:f32,i:!fir.array<5xi32>}>
  ! CHECK:         %[[VAL_14:.*]] = fir.coordinate_of %[[VAL_7]], %[[VAL_12b]] : (!fir.ref<!fir.type<_QMm_struct_ctorTt_array{x:f32,i:!fir.array<5xi32>}>>, !fir.field) -> !fir.ref<!fir.array<5xi32>>
  ! CHECK:         %[[VAL_15:.*]] = arith.constant 0 : index
  ! CHECK:         %[[VAL_16:.*]] = arith.constant 1 : index
  ! CHECK:         %[[VAL_17:.*]] = arith.constant 4 : index
  ! CHECK:         fir.do_loop %[[VAL_18:.*]] = %[[VAL_15]] to %[[VAL_17]] step %[[VAL_16]] {
  ! CHECK:           %[[VAL_19:.*]] = fir.coordinate_of %[[VAL_14]], %[[VAL_18]] : (!fir.ref<!fir.array<5xi32>>, index) -> !fir.ref<i32>
  ! CHECK:           %[[VAL_20:.*]] = fir.coordinate_of %[[VAL_13]], %[[VAL_18]] : (!fir.ref<!fir.array<5xi32>>, index) -> !fir.ref<i32>
  ! CHECK:           %[[VAL_21:.*]] = fir.load %[[VAL_20]] : !fir.ref<i32>
  ! CHECK:           fir.store %[[VAL_21]] to %[[VAL_19]] : !fir.ref<i32>
  ! CHECK:         }
    call print_nested(t_nested(x=x, dt=d))
  end subroutine

  subroutine print_simple(t)
    type(t_simple) :: t
    print *, t%x
  end subroutine
  subroutine print_char_scalar(t)
    type(t_char_scalar) :: t
    print *, t%x, t%c
  end subroutine
  subroutine print_simple_array(t)
    type(t_array) :: t
    print *, t%x, t%i
  end subroutine
  subroutine print_char_array(t)
    type(t_char_array) :: t
    print *, t%x, t%c
  end subroutine
  subroutine print_ptr(t)
    type(t_ptr) :: t
    print *, t%x, t%p
  end subroutine
  subroutine print_nested(t)
    type(t_nested) :: t
    print *, t%x, t%dt%x, t%dt%i
  end subroutine

end module

  use m_struct_ctor
  integer, target :: i(4,3) = reshape([1,2,3,4,5,6,7,8,9,10,11,12], [4,3])
  call test_simple(42.)
  call test_char_scalar(42.)
  call test_simple_array(42., [1,2,3,4,5])
  call test_char_array(42., ["abc", "def", "geh", "ijk", "lmn"])
  call test_ptr(42., i)
  call test_nested(42., t_array(x=43., i=[5,6,7,8,9]))
end
