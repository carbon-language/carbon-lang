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

module m
  type t1
  end type
  type t3
  end type
  interface
    subroutine s1(x)
      !ERROR: 't1' from host is not accessible
      import :: t1
      type(t1) :: x
      integer :: t1
    end subroutine
    subroutine s2()
      !ERROR: 't2' not found in host scope
      import :: t2
    end subroutine
    subroutine s3(x, y)
      !ERROR: Derived type 't1' not found
      type(t1) :: x, y
    end subroutine
    subroutine s4(x, y)
      !ERROR: 't3' from host is not accessible
      import, all
      type(t1) :: x
      type(t3) :: y
      integer :: t3
    end subroutine
  end interface
contains
  subroutine s5()
  end
  subroutine s6()
    import, only: s5
    implicit none(external)
    call s5()
  end
  subroutine s7()
    import, only: t1
    implicit none(external)
    !ERROR: 's5' is an external procedure without the EXTERNAL attribute in a scope with IMPLICIT NONE(EXTERNAL)
    call s5()
  end
end module
