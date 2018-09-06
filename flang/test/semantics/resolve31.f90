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

subroutine s1
  integer :: t0
  !ERROR: 't0' is not a derived type
  type(t0) :: x
  type :: t1
  end type
  type, extends(t1) :: t2
  end type
  !ERROR: Derived type 't3' not found
  type, extends(t3) :: t4
  end type
  !ERROR: 't0' is not a derived type
  type, extends(t0) :: t5
  end type
end subroutine

module m1
  type t0
  end type
end
module m2
  type t
  end type
end
module m3
  type t0
  end type
end
subroutine s2
  use m1
  use m2, t0 => t
  use m3
  !ERROR: Reference to 't0' is ambiguous
  type, extends(t0) :: t1
  end type
end subroutine

module m4
  type :: t1
    private
    sequence
    !ERROR: PRIVATE may not appear more than once in derived type components
    private
  end type
  !ERROR: A sequence type may not have the EXTENDS attribute
  type, extends(t1) :: t2
    sequence
    integer i
  end type
  !ERROR: A sequence type may not have a CONTAINS statement
  type :: t3
    sequence
    integer i
  contains
  end type
contains
  subroutine s3
    type :: t1
      !ERROR: PRIVATE is only allowed in a derived type that is in a module
      private
    contains
      !ERROR: PRIVATE is only allowed in a derived type that is in a module
      private
    end type
  end
end
