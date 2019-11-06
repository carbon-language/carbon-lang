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

!OPTIONS: -flogical-abbreviations -fxor-operator

! Like m4 in resolve63 but compiled with different options.
! Alternate operators are enabled so treat these as intrinsic.
module m4
contains
  subroutine s1(x, y, z)
    logical :: x
    real :: y, z
    !ERROR: Operands of .AND. must be LOGICAL; have REAL(4) and REAL(4)
    x = y .a. z
    !ERROR: Operands of .OR. must be LOGICAL; have REAL(4) and REAL(4)
    x = y .o. z
    !ERROR: Operand of .NOT. must be LOGICAL; have REAL(4)
    x = .n. y
    !ERROR: Operands of .NEQV. must be LOGICAL; have REAL(4) and REAL(4)
    x = y .xor. z
    !ERROR: Operands of .NEQV. must be LOGICAL; have REAL(4) and REAL(4)
    x = y .x. y
  end
end

! Like m4 in resolve63 but compiled with different options.
! Alternate operators are enabled so treat .A. as .AND.
module m5
  interface operator(.A.)
    logical function f1(x, y)
      integer, intent(in) :: x, y
    end
  end interface
  interface operator(.and.)
    logical function f2(x, y)
      real, intent(in) :: x, y
    end
  end interface
contains
  subroutine s1(x, y, z)
    logical :: x
    complex :: y, z
    !ERROR: No user-defined or intrinsic .A. operator matches operand types COMPLEX(4) and COMPLEX(4)
    x = y .and. z
    !ERROR: No user-defined or intrinsic .A. operator matches operand types COMPLEX(4) and COMPLEX(4)
    x = y .a. z
  end
end
