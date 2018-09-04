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

! Derived type parameters

module m
  !ERROR: Duplicate type parameter name: 'a'
  type t1(a, b, a)
    integer, kind :: a
    integer(8), len :: b
  end type
  !ERROR: No definition found for type parameter 'b'
  type t2(a, b, c)
    integer, kind :: a
    integer, len :: c
  end type
  !ERROR: 'b' is not defined as a type parameter
  type t3(a, b)
    integer, kind :: a
    integer :: b
  end type
  type t4(a)
    integer, kind :: a
    !ERROR: 'd' is not a type parameter of this derived type
    integer(8), len :: d
  end type
end module
