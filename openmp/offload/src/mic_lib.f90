!
!//===----------------------------------------------------------------------===//
!//
!//                     The LLVM Compiler Infrastructure
!//
!// This file is dual licensed under the MIT and the University of Illinois Open
!// Source Licenses. See LICENSE.txt for details.
!//
!//===----------------------------------------------------------------------===//
!


! **********************************************************************************
! * This file is intended to support the Intel(r) Many Integrated Core Architecture.
! **********************************************************************************
! free form Fortran source - should be named .f90
! lines are longer than 72 characters

module mic_lib
use, intrinsic :: iso_c_binding

integer, parameter:: target_mic=2
integer, parameter:: default_target_type=target_mic
integer, parameter:: default_target_number=0

enum, bind(C)
    enumerator :: OFFLOAD_SUCCESS  = 0
    enumerator :: OFFLOAD_DISABLED          ! offload is disabled
    enumerator :: OFFLOAD_UNAVAILABLE       ! card is not available
    enumerator :: OFFLOAD_OUT_OF_MEMORY     ! not enough memory on device
    enumerator :: OFFLOAD_PROCESS_DIED      ! target process has died
    enumerator :: OFFLOAD_ERROR             ! unspecified error
end enum

type, bind (C) :: offload_status
    integer(kind=c_int)    ::  result          = OFFLOAD_DISABLED
    integer(kind=c_int)    ::  device_number   = -1
    integer(kind=c_size_t) ::  data_sent       = 0
    integer(kind=c_size_t) ::  data_received   = 0
end type offload_status

interface
function offload_number_of_devices ()                                  &
           bind (C, name = "_Offload_number_of_devices")
!dec$ attributes default :: offload_number_of_devices
  import :: c_int
  integer (kind=c_int)        :: offload_number_of_devices
!dec$ attributes offload:mic :: offload_number_of_devices
!dir$ attributes known_intrinsic ::  offload_number_of_devices
end function offload_number_of_devices

function offload_signaled(target_number, signal)                       &
           bind (C, name = "_Offload_signaled")
!dec$ attributes default :: offload_signaled
  import :: c_int, c_int64_t
  integer (kind=c_int) :: offload_signaled
  integer (kind=c_int), value :: target_number
  integer (kind=c_int64_t), value :: signal
!dec$ attributes offload:mic :: offload_signaled
end function offload_signaled

subroutine offload_report(val)                                         &
           bind (C, name = "_Offload_report")
!dec$ attributes default :: offload_report
  import :: c_int
  integer (kind=c_int), value :: val
!dec$ attributes offload:mic :: offload_report
end subroutine offload_report

function offload_get_device_number()                                   &
           bind (C, name = "_Offload_get_device_number")
!dec$ attributes default :: offload_get_device_number
  import :: c_int
  integer (kind=c_int)        :: offload_get_device_number
!dec$ attributes offload:mic :: offload_get_device_number
end function offload_get_device_number

function offload_get_physical_device_number()                          &
           bind (C, name = "_Offload_get_physical_device_number")
!dec$ attributes default :: offload_get_physical_device_number
  import :: c_int
  integer (kind=c_int)        :: offload_get_physical_device_number
!dec$ attributes offload:mic :: offload_get_physical_device_number
end function offload_get_physical_device_number

! OpenMP API wrappers

subroutine omp_set_num_threads_target (target_type,                    &
                                       target_number,                  &
                                       num_threads)                    &
           bind (C, name = "omp_set_num_threads_target")
  import :: c_int
  integer (kind=c_int), value :: target_type, target_number, num_threads
end subroutine omp_set_num_threads_target

function omp_get_max_threads_target (target_type,                      &
                                     target_number)                    &
         bind (C, name = "omp_get_max_threads_target")
  import :: c_int
  integer (kind=c_int)        :: omp_get_max_threads_target
  integer (kind=c_int), value :: target_type, target_number
end function omp_get_max_threads_target

function omp_get_num_procs_target (target_type,                        &
                                   target_number)                      &
         bind (C, name = "omp_get_num_procs_target")
  import :: c_int
  integer (kind=c_int)        :: omp_get_num_procs_target
  integer (kind=c_int), value :: target_type, target_number
end function omp_get_num_procs_target

subroutine omp_set_dynamic_target (target_type,                        &
                                   target_number,                      &
                                   num_threads)                        &
           bind (C, name = "omp_set_dynamic_target")
  import :: c_int
  integer (kind=c_int), value :: target_type, target_number, num_threads
end subroutine omp_set_dynamic_target

function omp_get_dynamic_target (target_type,                          &
                                 target_number)                        &
         bind (C, name = "omp_get_dynamic_target")
  import :: c_int
  integer (kind=c_int)        :: omp_get_dynamic_target
  integer (kind=c_int), value :: target_type, target_number
end function omp_get_dynamic_target

subroutine omp_set_nested_target (target_type,                         &
                                  target_number,                       &
                                  nested)                              &
           bind (C, name = "omp_set_nested_target")
  import :: c_int
  integer (kind=c_int), value :: target_type, target_number, nested
end subroutine omp_set_nested_target

function omp_get_nested_target (target_type,                           &
                                target_number)                         &
         bind (C, name = "omp_get_nested_target")
  import :: c_int
  integer (kind=c_int)        :: omp_get_nested_target
  integer (kind=c_int), value :: target_type, target_number
end function omp_get_nested_target

subroutine omp_set_schedule_target (target_type,                       &
                                    target_number,                     &
                                    kind,                              &
                                    modifier)                          &
           bind (C, name = "omp_set_schedule_target")
  import :: c_int
  integer (kind=c_int), value :: target_type, target_number, kind, modifier
end subroutine omp_set_schedule_target

subroutine omp_get_schedule_target (target_type,                       &
                                    target_number,                     &
                                    kind,                              &
                                    modifier)                          &
           bind (C, name = "omp_get_schedule_target")
  import :: c_int, c_intptr_t
  integer (kind=c_int), value :: target_type, target_number
  integer (kind=c_intptr_t), value :: kind, modifier
end subroutine omp_get_schedule_target

! lock API functions

subroutine omp_init_lock_target (target_type,                          &
                                 target_number,                        &
                                 lock)                                 &
           bind (C, name = "omp_init_lock_target")
  import :: c_int, c_intptr_t
  !dir$ attributes known_intrinsic ::  omp_init_lock_target
  integer (kind=c_int), value :: target_type, target_number
  integer (kind=c_intptr_t), value :: lock
end subroutine omp_init_lock_target

subroutine omp_destroy_lock_target (target_type,                       &
                                    target_number,                     &
                                    lock)                              &
           bind (C, name = "omp_destroy_lock_target")
  import :: c_int, c_intptr_t
  !dir$ attributes known_intrinsic ::  omp_destroy_lock_target
  integer (kind=c_int), value :: target_type, target_number
  integer (kind=c_intptr_t), value :: lock
end subroutine omp_destroy_lock_target

subroutine omp_set_lock_target (target_type,                           &
                                target_number,                         &
                                lock)                                  &
           bind (C, name = "omp_set_lock_target")
  import :: c_int, c_intptr_t
  !dir$ attributes known_intrinsic ::  omp_set_lock_target
  integer (kind=c_int), value :: target_type, target_number
  integer (kind=c_intptr_t), value :: lock
end subroutine omp_set_lock_target

subroutine omp_unset_lock_target (target_type,                         &
                                  target_number,                       &
                                  lock)                                &
           bind (C, name = "omp_unset_lock_target")
  import :: c_int, c_intptr_t
  !dir$ attributes known_intrinsic ::  omp_unset_lock_target
  integer (kind=c_int), value :: target_type, target_number
  integer (kind=c_intptr_t), value :: lock
end subroutine omp_unset_lock_target

function omp_test_lock_target (target_type,                            &
                               target_number,                          &
                               lock)                                   &
           bind (C, name = "omp_test_lock_target")
  import :: c_int, c_intptr_t
  integer (kind=c_int)        :: omp_test_lock_target
  integer (kind=c_int), value :: target_type, target_number
  integer (kind=c_intptr_t), value :: lock
end function omp_test_lock_target

! nested lock API functions

subroutine omp_init_nest_lock_target (target_type,                     &
                                      target_number,                   &
                                      lock)                            &
           bind (C, name = "omp_init_nest_lock_target")
  import :: c_int, c_intptr_t
  integer (kind=c_int), value :: target_type, target_number
  integer (kind=c_intptr_t), value :: lock
end subroutine omp_init_nest_lock_target

subroutine omp_destroy_nest_lock_target (target_type,                  &
                                         target_number,                &
                                         lock)                         &
           bind (C, name = "omp_destroy_nest_lock_target")
  import :: c_int, c_intptr_t
  integer (kind=c_int), value :: target_type, target_number
  integer (kind=c_intptr_t), value :: lock
end subroutine omp_destroy_nest_lock_target

subroutine omp_set_nest_lock_target (target_type,                      &
                                     target_number,                    &
                                     lock)                             &
           bind (C, name = "omp_set_nest_lock_target")
  import :: c_int, c_intptr_t
  integer (kind=c_int), value :: target_type, target_number
  integer (kind=c_intptr_t), value :: lock
end subroutine omp_set_nest_lock_target

subroutine omp_unset_nest_lock_target (target_type,                    &
                                       target_number,                  &
                                       lock)                           &
           bind (C, name = "omp_unset_nest_lock_target")
  import :: c_int, c_intptr_t
  integer (kind=c_int), value :: target_type, target_number
  integer (kind=c_intptr_t), value :: lock
end subroutine omp_unset_nest_lock_target

function omp_test_nest_lock_target (target_type,                       &
                                    target_number,                     &
                                    lock)                              &
           bind (C, name = "omp_test_nest_lock_target")
  import :: c_int, c_intptr_t
  integer (kind=c_int)        :: omp_test_nest_lock_target
  integer (kind=c_int), value :: target_type, target_number
  integer (kind=c_intptr_t), value :: lock
end function omp_test_nest_lock_target

! kmp API functions

subroutine kmp_set_stacksize_target (target_type,                      &
                                     target_number,                    &
                                     size)                             &
           bind (C, name = "kmp_set_stacksize_target")
  import :: c_int
  integer (kind=c_int), value :: target_type, target_number, size
end subroutine kmp_set_stacksize_target

function kmp_get_stacksize_target (target_type,                        &
                                   target_number)                      &
         bind (C, name = "kmp_get_stacksize_target")
  import :: c_int
  integer (kind=c_int)        :: kmp_get_stacksize_target
  integer (kind=c_int), value :: target_type, target_number
end function kmp_get_stacksize_target

subroutine kmp_set_stacksize_s_target (target_type,                    &
                                       target_number,                  &
                                       size)                           &
           bind (C, name = "kmp_set_stacksize_s_target")
  import :: c_int
  integer (kind=c_int), value :: target_type, target_number, size
end subroutine kmp_set_stacksize_s_target

function kmp_get_stacksize_s_target (target_type,                      &
                                     target_number)                    &
         bind (C, name = "kmp_get_stacksize_s_target")
  import :: c_int
  integer (kind=c_int)        :: kmp_get_stacksize_s_target
  integer (kind=c_int), value :: target_type, target_number
end function kmp_get_stacksize_s_target

subroutine kmp_set_blocktime_target (target_type,                      &
                                     target_number,                    &
                                     time)                             &
           bind (C, name = "kmp_set_blocktime_target")
  import :: c_int
  integer (kind=c_int), value :: target_type, target_number, time
end subroutine kmp_set_blocktime_target

function kmp_get_blocktime_target (target_type,                        &
                                   target_number)                      &
         bind (C, name = "kmp_get_blocktime_target")
  import :: c_int
  integer (kind=c_int)        :: kmp_get_blocktime_target
  integer (kind=c_int), value :: target_type, target_number
end function kmp_get_blocktime_target

subroutine kmp_set_library_serial_target (target_type,                 &
                                          target_number)               &
           bind (C, name = "kmp_set_library_serial_target")
  import :: c_int
  integer (kind=c_int), value :: target_type, target_number
end subroutine kmp_set_library_serial_target

subroutine kmp_set_library_turnaround_target (target_type,             &
                                              target_number)           &
           bind (C, name = "kmp_set_library_turnaround_target")
  import :: c_int
  integer (kind=c_int), value :: target_type, target_number
end subroutine kmp_set_library_turnaround_target

subroutine kmp_set_library_throughput_target (target_type,             &
                                              target_number)           &
           bind (C, name = "kmp_set_library_throughput_target")
  import :: c_int
  integer (kind=c_int), value :: target_type, target_number
end subroutine kmp_set_library_throughput_target

subroutine kmp_set_library_target (target_type,                        &
                                   target_number,                      &
                                   mode)                               &
           bind (C, name = "kmp_set_library_target")
  import :: c_int
  integer (kind=c_int), value :: target_type, target_number, mode
end subroutine kmp_set_library_target

function kmp_get_library_target (target_type,                          &
                                 target_number)                        &
         bind (C, name = "kmp_get_library_target")
  import :: c_int
  integer (kind=c_int)        :: kmp_get_library_target
  integer (kind=c_int), value :: target_type, target_number
end function kmp_get_library_target

subroutine kmp_set_defaults_target (target_type,                       &
                                    target_number,                     &
                                    defaults)                          &
           bind (C, name = "kmp_set_defaults_target")
  import :: c_int, c_char
 character (kind=c_char) :: defaults(*)
 integer (kind=c_int), value :: target_type, target_number
end subroutine kmp_set_defaults_target

! affinity API functions

subroutine kmp_create_affinity_mask_target (target_type,               &
                                            target_number,             &
                                            mask)                      &
           bind (C, name = "kmp_create_affinity_mask_target")
  import :: c_int, c_intptr_t
  integer (kind=c_int), value :: target_type, target_number
  integer (kind=c_intptr_t), value :: mask
end subroutine kmp_create_affinity_mask_target

subroutine kmp_destroy_affinity_mask_target (target_type,              &
                                             target_number,            &
                                             mask)                     &
           bind (C, name = "kmp_destroy_affinity_mask_target")
  import :: c_int, c_intptr_t
  integer (kind=c_int), value :: target_type, target_number
  integer (kind=c_intptr_t), value :: mask
end subroutine kmp_destroy_affinity_mask_target

function kmp_set_affinity_target (target_type,                         &
                                  target_number,                       &
                                  mask)                                &
           bind (C, name = "kmp_set_affinity_target")
  import :: c_int, c_intptr_t
  integer (kind=c_int)        :: kmp_set_affinity_target
  integer (kind=c_int), value :: target_type, target_number
  integer (kind=c_intptr_t), value :: mask
end function kmp_set_affinity_target

function kmp_get_affinity_target (target_type,                         &
                                  target_number,                       &
                                  mask)                                &
           bind (C, name = "kmp_get_affinity_target")
  import :: c_int, c_intptr_t
  integer (kind=c_int)        :: kmp_get_affinity_target
  integer (kind=c_int), value :: target_type, target_number
  integer (kind=c_intptr_t), value :: mask
end function kmp_get_affinity_target

function kmp_get_affinity_max_proc_target (target_type,                &
                                           target_number)              &
           bind (C, name = "kmp_get_affinity_max_proc_target")
  import :: c_int
  integer (kind=c_int)        :: kmp_get_affinity_max_proc_target
  integer (kind=c_int), value :: target_type, target_number
end function kmp_get_affinity_max_proc_target

function kmp_set_affinity_mask_proc_target (target_type,               &
                                            target_number,             &
                                            proc,                      &
                                            mask)                      &
           bind (C, name = "kmp_set_affinity_mask_proc_target")
  import :: c_int, c_intptr_t
  integer (kind=c_int)        :: kmp_set_affinity_mask_proc_target
  integer (kind=c_int), value :: target_type, target_number, proc
  integer (kind=c_intptr_t), value :: mask
end function kmp_set_affinity_mask_proc_target

function kmp_unset_affinity_mask_proc_target (target_type,             &
                                              target_number,           &
                                              proc,                    &
                                              mask)                    &
           bind (C, name = "kmp_unset_affinity_mask_proc_target")
  import :: c_int, c_intptr_t
  integer (kind=c_int)        :: kmp_unset_affinity_mask_proc_target
  integer (kind=c_int), value :: target_type, target_number, proc
  integer (kind=c_intptr_t), value :: mask
end function kmp_unset_affinity_mask_proc_target

function kmp_get_affinity_mask_proc_target (target_type,               &
                                            target_number,             &
                                            proc,                      &
                                            mask)                      &
           bind (C, name = "kmp_get_affinity_mask_proc_target")
  import :: c_int, c_intptr_t
  integer (kind=c_int)        :: kmp_get_affinity_mask_proc_target
  integer (kind=c_int), value :: target_type, target_number, proc
  integer (kind=c_intptr_t), value :: mask
end function kmp_get_affinity_mask_proc_target

end interface
end module mic_lib
