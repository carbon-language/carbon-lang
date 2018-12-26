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

module m1
  type t1
  contains
    procedure, nopass :: s1
    !ERROR: Binding name 's2' not found in this derived type
    generic :: g1 => s2
  end type
  type t2
    integer :: s1
  contains
    !ERROR: 's1' is not the name of a specific binding of this type
    generic :: g2 => s1
  end type
contains
  subroutine s1
  end
end

module m2
  type :: t3
  contains
    private
    procedure, nopass :: s3
    generic, public :: g3 => s3
    generic :: h3 => s3
  end type
contains
  subroutine s3(i)
  end
end

module m3
  use m2
  type, extends(t3) :: t4
  contains
    procedure, nopass :: s4
    procedure, nopass :: s5
    !ERROR: 'g3' does not have the same accessibility as its previous declaration
    generic, private :: g3 => s4
    !ERROR: 'h3' does not have the same accessibility as its previous declaration
    generic, public :: h3 => s4
    generic :: i3 => s4
    !ERROR: 'i3' does not have the same accessibility as its previous declaration
    generic, private :: i3 => s5
  end type
  type :: t5
  contains
    private
    procedure, nopass :: s3
    procedure, nopass :: s4
    procedure, nopass :: s5
    generic :: g5 => s3, s4
    !ERROR: 'g5' does not have the same accessibility as its previous declaration
    generic, public :: g5 => s5
  end type
contains
  subroutine s4(r)
  end
  subroutine s5(z)
    complex :: z
  end
end
