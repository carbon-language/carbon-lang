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
! CHECK: END BLOCK DATA name mismatch
! CHECK: mismatched from here
! CHECK: END FUNCTION name mismatch
! CHECK: END SUBROUTINE name mismatch
! CHECK: END PROGRAM name mismatch
! CHECK: END SUBMODULE name mismatch
! CHECK: END TYPE name mismatch
! CHECK: END MODULE name mismatch
! CHECK: INTERFACE generic-name
! CHECK: END MODULE PROCEDURE name mismatch

block data t1
end block data t2

function t3
end function t4

subroutine t9
end subroutine t10

program t13
end program t14

submodule (mod) t15
end submodule t16

module t5
  interface t7
  end interface t8
  type t17
  end type t18
contains
  module procedure t11
  end procedure t12
end module mox
