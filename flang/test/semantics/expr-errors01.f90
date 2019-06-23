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

! C1003 - can't parenthesize function call returning procedure pointer
module m1
  type :: dt
    procedure(frpp), pointer, nopass :: pp
  end type dt
 contains
  subroutine boring
  end subroutine boring
  function frpp
    procedure(boring), pointer :: frpp
    frpp => boring
  end function frpp
  subroutine tests
    procedure(boring), pointer :: mypp
    type(dt) :: dtinst
    mypp => boring ! legal
    mypp => (boring) ! legal, not a function reference
    !ERROR: A function reference that returns a procedure pointer may not be parenthesized.
    mypp => (frpp()) ! C1003
    mypp => frpp() ! legal, not parenthesized
    dtinst%pp => frpp
    mypp => dtinst%pp() ! legal
    !ERROR: A function reference that returns a procedure pointer may not be parenthesized.
    mypp => (dtinst%pp())
  end subroutine tests
end module m1
