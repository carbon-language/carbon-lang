! Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
!
! Licensed under the Apache License, Version 2.0 (the "License");
! you may not use this file except in compliance with the License.
! You may obtain a copy of the License at
!
!     http://www.apache.org/licenses/LICENSE-2.0
!
! Unless required by applicable law or agreed to in writing, software
! distributed under the License is distributed on an "AS IS" BASIS,
! WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
! See the License for the specific language governing permissions and
! limitations under the License.
!
! C1141
! A reference to the procedure IEEE_GET_FLAG, IEEE_SET_HALTING_MODE, or
! IEEE_GET_HALTING_MODE from the intrinsic module IEEE_EXCEPTIONS, shall not
! appear within a DO CONCURRENT construct.
!
! C1137
! An image control statement shall not appear within a DO CONCURRENT construct.
!
! C1136 
! A RETURN statement shall not appear within a DO CONCURRENT construct.
!
! (11.1.7.5), paragraph 4
! In a DO CONCURRENT, can't have an i/o statement with an ADVANCE= specifier

subroutine do_concurrent_test1(i,n)
  implicit none
  integer :: i, n
  do 10 concurrent (i = 1:n)
!ERROR: image control statement not allowed in DO CONCURRENT
     SYNC ALL
!ERROR: image control statement not allowed in DO CONCURRENT
     SYNC IMAGES (*)
!ERROR: image control statement not allowed in DO CONCURRENT
     SYNC MEMORY
!ERROR: RETURN not allowed in DO CONCURRENT
     return
10 continue
end subroutine do_concurrent_test1

subroutine do_concurrent_test2(i,j,n,flag)
  use ieee_exceptions
  use iso_fortran_env, only: team_type
  implicit none
  integer :: i, n, flag, flag2
  logical :: halting
  type(team_type) :: j
  do concurrent (i = 1:n)
!ERROR: image control statement not allowed in DO CONCURRENT
    sync team (j)
    change team (j)
      critical
!ERROR: call to impure procedure in DO CONCURRENT not allowed
!ERROR: IEEE_GET_FLAG not allowed in DO CONCURRENT
        call ieee_get_flag(flag, flag2)
!ERROR: call to impure procedure in DO CONCURRENT not allowed
!ERROR: IEEE_GET_HALTING_MODE not allowed in DO CONCURRENT
        call ieee_get_halting_mode(flag, halting)
!ERROR: IEEE_SET_HALTING_MODE not allowed in DO CONCURRENT
        call ieee_set_halting_mode(flag, halting)
!ERROR: image control statement not allowed in DO CONCURRENT
      end critical
!ERROR: image control statement not allowed in DO CONCURRENT
    end team
!ERROR: ADVANCE specifier not allowed in DO CONCURRENT
    write(*,'(a35)',advance='no')
  end do
end subroutine do_concurrent_test2

subroutine s1()
  use iso_fortran_env
  type(event_type) :: x
  do concurrent (i = 1:n)
!ERROR: image control statement not allowed in DO CONCURRENT
    event post (x)
  end do
end subroutine s1

subroutine s2()
  use iso_fortran_env
  type(event_type) :: x
  do concurrent (i = 1:n)
!ERROR: image control statement not allowed in DO CONCURRENT
    event wait (x)
  end do
end subroutine s2

subroutine s3()
  use iso_fortran_env
  type(team_type) :: t

  do concurrent (i = 1:n)
!ERROR: image control statement not allowed in DO CONCURRENT
    form team(1, t)
  end do
end subroutine s3

subroutine s4()
  use iso_fortran_env
  type(lock_type) :: l

  do concurrent (i = 1:n)
!ERROR: image control statement not allowed in DO CONCURRENT
    lock(l)
!ERROR: image control statement not allowed in DO CONCURRENT
    unlock(l)
  end do
end subroutine s4

subroutine s5()
  use iso_fortran_env
  type(lock_type) :: l

  do concurrent (i = 1:n)
!ERROR: image control statement not allowed in DO CONCURRENT
    lock(l)
!ERROR: image control statement not allowed in DO CONCURRENT
    unlock(l)
  end do
end subroutine s5
