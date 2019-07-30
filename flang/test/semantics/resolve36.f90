! Copyright (c) 2018-2019, NVIDIA CORPORATION.  All rights reserved.
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

module m1
  interface
    module subroutine sub1(arg1)
      integer, intent(inout) :: arg1
    end subroutine
    module integer function fun1()
    end function
  end interface
  type t
  end type
  integer i
end module

submodule(m1) s1
contains
  !ERROR: 'missing1' was not declared a separate module procedure
  module procedure missing1
  end
  !ERROR: 'missing2' was not declared a separate module procedure
  module subroutine missing2
  end
  !ERROR: 't' was not declared a separate module procedure
  module procedure t
  end
  !ERROR: 'i' was not declared a separate module procedure
  module subroutine i
  end
end submodule

module m2
  interface
    module subroutine sub1(arg1)
      integer, intent(inout) :: arg1
    end subroutine
    module integer function fun1()
    end function
  end interface
  type t
  end type
  !ERROR: Declaration of 'i' conflicts with its use as module procedure
  integer i
contains
  !ERROR: 'missing1' was not declared a separate module procedure
  module procedure missing1
  end
  !ERROR: 'missing2' was not declared a separate module procedure
  module subroutine missing2
  end
  !ERROR: 't' is already declared in this scoping unit
  !ERROR: 't' was not declared a separate module procedure
  module procedure t
  end
  !ERROR: 'i' was not declared a separate module procedure
  module subroutine i
  end
end module

! Separate module procedure defined in same module as declared
module m3
  interface
    module subroutine sub
    end subroutine
  end interface
contains
  module procedure sub
  end procedure
end module
