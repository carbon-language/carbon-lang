! Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
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

! Test 15.5.2.9(5) dummy procedure POINTER requirements

module m

 contains

  subroutine s01(p)
    procedure(sin), pointer, intent(in) :: p
  end subroutine
  subroutine s02(p)
    procedure(sin), pointer :: p
  end subroutine

  function procptr()
    procedure(sin), pointer :: procptr
    procptr => cos
  end function

  subroutine test
    procedure(tan), pointer :: p
    p => tan
    call s01(p) ! ok
    call s01(procptr()) ! ok
    call s01(null()) ! ok
    call s01(null(p)) ! ok
    call s01(sin) ! ok
    call s02(p) ! ok
    ! ERROR: Effective argument associated with dummy procedure pointer must be a procedure pointer unless INTENT(IN)
    call s02(procptr())
    ! ERROR: Effective argument associated with dummy procedure pointer must be a procedure pointer unless INTENT(IN)
    call s02(null())
    ! ERROR: Effective argument associated with dummy procedure pointer must be a procedure pointer unless INTENT(IN)
    call s02(null(p))
    ! ERROR: Effective argument associated with dummy procedure pointer must be a procedure pointer unless INTENT(IN)
    call s02(sin)
  end subroutine

end module
