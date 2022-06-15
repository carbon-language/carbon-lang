!RUN: %flang_fc1 -emit-fir -fopenmp %s -o - | FileCheck %s --check-prefixes="FIRDialect,OMPDialect"
!RUN: %flang_fc1 -emit-fir -fopenmp %s -o - | fir-opt --cfg-conversion | fir-opt --fir-to-llvm-ir | FileCheck %s --check-prefixes="LLVMDialect,OMPDialect"

!===============================================================================
! Single construct
!===============================================================================

!FIRDialect-LABEL: func @_QPomp_single
!FIRDialect-SAME: (%[[x:.*]]: !fir.ref<i32> {fir.bindc_name = "x"})
subroutine omp_single(x)
  integer, intent(inout) :: x
  !OMPDialect: omp.parallel
  !$omp parallel
  !OMPDialect: omp.single
  !$omp single
    !FIRDialect: %[[xval:.*]] = fir.load %[[x]] : !fir.ref<i32>
    !FIRDialect: %[[res:.*]] = arith.addi %[[xval]], %{{.*}} : i32
    !FIRDialect: fir.store %[[res]] to %[[x]] : !fir.ref<i32>
    x = x + 12
  !OMPDialect: omp.terminator
  !$omp end single
  !OMPDialect: omp.terminator
  !$omp end parallel
end subroutine omp_single

!===============================================================================
! Single construct with nowait
!===============================================================================

!FIRDialect-LABEL: func @_QPomp_single_nowait
!FIRDialect-SAME: (%[[x:.*]]: !fir.ref<i32> {fir.bindc_name = "x"})
subroutine omp_single_nowait(x)
  integer, intent(inout) :: x
  !OMPDialect: omp.parallel
  !$omp parallel
  !OMPDialect: omp.single nowait
  !$omp single
    !FIRDialect: %[[xval:.*]] = fir.load %[[x]] : !fir.ref<i32>
    !FIRDialect: %[[res:.*]] = arith.addi %[[xval]], %{{.*}} : i32
    !FIRDialect: fir.store %[[res]] to %[[x]] : !fir.ref<i32>
    x = x + 12
  !OMPDialect: omp.terminator
  !$omp end single nowait
  !OMPDialect: omp.terminator
  !$omp end parallel
end subroutine omp_single_nowait

!===============================================================================
! Single construct with allocate
!===============================================================================

!FIRDialect-LABEL: func @_QPsingle_allocate
subroutine single_allocate()
  use omp_lib
  integer :: x
  !OMPDialect: omp.parallel {
  !$omp parallel
  !OMPDialect: omp.single allocate(
  !FIRDialect: %{{.+}} : i32 -> %{{.+}} : !fir.ref<i32>
  !LLVMDialect: %{{.+}} : i32 -> %{{.+}} : !llvm.ptr<i32>
  !OMPDialect: ) {
  !$omp single allocate(omp_high_bw_mem_alloc: x) private(x)
  !FIRDialect: arith.addi
  x = x + 12
  !OMPDialect: omp.terminator
  !$omp end single
  !OMPDialect: omp.terminator
  !$omp end parallel
end subroutine single_allocate
