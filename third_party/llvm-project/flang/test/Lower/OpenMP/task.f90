!RUN: %flang_fc1 -emit-fir -fopenmp %s -o - | FileCheck %s

!CHECK-LABEL: func @_QPomp_task_simple() {
subroutine omp_task_simple
  !CHECK: omp.task {
  !$omp task
  !CHECK: fir.call @_QPfoo() : () -> ()
  call foo()
  !CHECK: omp.terminator
  !$omp end task
end subroutine omp_task_simple

!===============================================================================
! `if` clause
!===============================================================================

!CHECK-LABEL: func @_QPomp_task_if(%{{.+}}) {
subroutine omp_task_if(bar)
  logical, intent(inout) :: bar
  !CHECK: omp.task if(%{{.+}}) {
  !$omp task if(bar)
  !CHECK: fir.call @_QPfoo() : () -> ()
  call foo()
  !CHECK: omp.terminator
  !$omp end task
end subroutine omp_task_if

!===============================================================================
! `final` clause
!===============================================================================

!CHECK-LABEL: func @_QPomp_task_final(%{{.+}}) {
subroutine omp_task_final(bar)
  logical, intent(inout) :: bar
  !CHECK: omp.task final(%{{.+}}) {
  !$omp task final(bar)
  !CHECK: fir.call @_QPfoo() : () -> ()
  call foo()
  !CHECK: omp.terminator
  !$omp end task
end subroutine omp_task_final

!===============================================================================
! `untied` clause
!===============================================================================

!CHECK-LABEL: func @_QPomp_task_untied() {
subroutine omp_task_untied()
  !CHECK: omp.task untied {
  !$omp task untied
  !CHECK: fir.call @_QPfoo() : () -> ()
  call foo()
  !CHECK: omp.terminator
  !$omp end task
end subroutine omp_task_untied

!===============================================================================
! `mergeable` clause
!===============================================================================

!CHECK-LABEL: func @_QPomp_task_mergeable() {
subroutine omp_task_mergeable()
  !CHECK: omp.task mergeable {
  !$omp task mergeable
  !CHECK: fir.call @_QPfoo() : () -> ()
  call foo()
  !CHECK: omp.terminator
  !$omp end task
end subroutine omp_task_mergeable

!===============================================================================
! `priority` clause
!===============================================================================

!CHECK-LABEL: func @_QPomp_task_priority(%{{.+}}) {
subroutine omp_task_priority(bar)
  integer, intent(inout) :: bar
  !CHECK: omp.task priority(%{{.+}}) {
  !$omp task priority(bar)
  !CHECK: fir.call @_QPfoo() : () -> ()
  call foo()
  !CHECK: omp.terminator
  !$omp end task
end subroutine omp_task_priority

!===============================================================================
! `allocate` clause
!===============================================================================

!CHECK-LABEL: func @_QPtask_allocate
subroutine task_allocate()
  use omp_lib
  integer :: x
  !CHECK: omp.task allocate(%{{.+}} : i32 -> %{{.+}} : !fir.ref<i32>) {
  !$omp task allocate(omp_high_bw_mem_alloc: x) private(x)
  !CHECK: arith.addi
  x = x + 12
  !CHECK: omp.terminator
  !$omp end task
end subroutine task_allocate

!===============================================================================
! `private` clause
!===============================================================================
!CHECK-LABEL: func @_QPtask_private
subroutine task_private
  type mytype
  integer :: x
  end type mytype

  !CHECK: %[[int_var:.+]] = fir.alloca i32
  !CHECK: %[[mytype_var:.+]] = fir.alloca !fir.type<_QFtask_privateTmytype{x:i32}>
  integer :: int_var
  type(mytype) :: mytype_var

  !CHECK: fir.call @_QPbar(%[[int_var]], %[[mytype_var]]) : (!fir.ref<i32>, !fir.ref<!fir.type<_QFtask_privateTmytype{x:i32}>>) -> ()
  call bar(int_var, mytype_var)

  !CHECK: omp.task {
  !$omp task private(int_var, mytype_var)
  !CHECK: %[[int_var_private:.+]] = fir.alloca i32
  !CHECK: %[[mytype_var_private:.+]] = fir.alloca !fir.type<_QFtask_privateTmytype{x:i32}>

  !CHECK: fir.call @_QPbar(%[[int_var_private]], %[[mytype_var_private]]) : (!fir.ref<i32>, !fir.ref<!fir.type<_QFtask_privateTmytype{x:i32}>>) -> ()
  call bar(int_var, mytype_var)
  !CHECK: omp.terminator
  !$omp end task
end subroutine task_private

!===============================================================================
! `firstprivate` clause
!===============================================================================
!CHECK-LABEL: func @_QPtask_firstprivate
subroutine task_firstprivate
  type mytype
  integer :: x
  end type mytype

  !CHECK: %[[int_var:.+]] = fir.alloca i32
  !CHECK: %[[mytype_var:.+]] = fir.alloca !fir.type<_QFtask_firstprivateTmytype{x:i32}>
  integer :: int_var
  type(mytype) :: mytype_var

  !CHECK: fir.call @_QPbaz(%[[int_var]], %[[mytype_var]]) : (!fir.ref<i32>, !fir.ref<!fir.type<_QFtask_firstprivateTmytype{x:i32}>>) -> ()
  call baz(int_var, mytype_var)

  !CHECK: omp.task {
  !$omp task firstprivate(int_var, mytype_var)
  !CHECK: %[[int_var_firstprivate:.+]] = fir.alloca i32
  !CHECK: %[[int_var_load:.+]] = fir.load %[[int_var]] : !fir.ref<i32>
  !CHECK: fir.store %[[int_var_load]] to %[[int_var_firstprivate]] : !fir.ref<i32>
  !CHECK: %[[mytype_var_firstprivate:.+]] = fir.alloca !fir.type<_QFtask_firstprivateTmytype{x:i32}>
  !CHECK: %[[mytype_var_load:.+]] = fir.load %[[mytype_var]] : !fir.ref<!fir.type<_QFtask_firstprivateTmytype{x:i32}>>
  !CHECK: fir.store %[[mytype_var_load]] to %[[mytype_var_firstprivate]]
  !CHECK: fir.call @_QPbaz(%[[int_var_firstprivate]], %[[mytype_var_firstprivate]]) : (!fir.ref<i32>, !fir.ref<!fir.type<_QFtask_firstprivateTmytype{x:i32}>>) -> ()
  call baz(int_var, mytype_var)
  !CHECK: omp.terminator
  !$omp end task
end subroutine task_firstprivate

!===============================================================================
! Multiple clauses
!===============================================================================

!CHECK-LABEL: func @_QPtask_multiple_clauses
subroutine task_multiple_clauses()
  use omp_lib

  !CHECK: %[[x:.+]] = fir.alloca i32 {bindc_name = "x", uniq_name = "_QFtask_multiple_clausesEx"}
  !CHECK: %[[y:.+]] = fir.alloca i32 {bindc_name = "y", uniq_name = "_QFtask_multiple_clausesEy"}
  !CHECK: %[[z:.+]] = fir.alloca i32 {bindc_name = "z", uniq_name = "_QFtask_multiple_clausesEz"}
  integer :: x, y, z
  logical :: buzz

  !CHECK: omp.task if(%{{.+}}) final(%{{.+}}) untied mergeable priority(%{{.+}}) allocate(%{{.+}} : i32 -> %{{.+}} : !fir.ref<i32>) {
  !$omp task if(buzz) final(buzz) untied mergeable priority(z) allocate(omp_high_bw_mem_alloc: x) private(x) firstprivate(y)

  !CHECK: %[[x_priv:.+]] = fir.alloca i32
  !CHECK: %[[y_priv:.+]] = fir.alloca i32
  !CHECK: %[[y_load:.+]] = fir.load %[[y]] : !fir.ref<i32>
  !CHECK: fir.store %[[y_load]] to %[[y_priv]] : !fir.ref<i32>

  !CHECK: arith.addi
  x = x + 12
  !CHECK: arith.subi
  y = y - 12

  !CHECK: omp.terminator
  !$omp end task
end subroutine task_multiple_clauses
