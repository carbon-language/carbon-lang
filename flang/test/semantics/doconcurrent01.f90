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

! RUN: ${F18} -funparse-with-symbols %s 2>&1 | ${FileCheck} %s
! CHECK: image control statement not allowed in DO CONCURRENT
! CHECK: RETURN not allowed in DO CONCURRENT
! CHECK: call to impure subroutine in DO CONCURRENT not allowed
! CHECK: IEEE_GET_FLAG not allowed in DO CONCURRENT
! CHECK: ADVANCE specifier not allowed in DO CONCURRENT
! CHECK: SYNC ALL
! CHECK: SYNC IMAGES

module ieee_exceptions
  interface
     subroutine ieee_get_flag(i, j)
       integer :: i, j
     end subroutine ieee_get_flag
  end interface
end module ieee_exceptions

subroutine do_concurrent_test1(i,n)
  implicit none
  integer :: i, n
  do 10 concurrent (i = 1:n)
     SYNC ALL
     SYNC IMAGES (*)
     return
10 continue
end subroutine do_concurrent_test1

subroutine do_concurrent_test2(i,j,n,flag)
  use ieee_exceptions
  implicit none
  integer :: i, j, n, flag, flag2
  do concurrent (i = 1:n)
    change team (j)
      call ieee_get_flag(flag, flag2)
    end team
    write(*,'(a35)',advance='no')
  end do
end subroutine do_concurrent_test2
