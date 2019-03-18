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

module m1
  !ERROR: Logical constant '.true.' may not be used as a defined operator
  interface operator(.TRUE.)
  end interface
  !ERROR: Logical constant '.false.' may not be used as a defined operator
  generic :: operator(.false.) => bar
end

module m2
  interface operator(+)
    module procedure foo
  end interface
  interface operator(.foo.)
    module procedure foo
  end interface
  interface operator(.ge.)
    module procedure bar
  end interface
contains
  integer function foo(x, y)
    logical, intent(in) :: x, y
    foo = 0
  end
  logical function bar(x, y)
    complex, intent(in) :: x, y
    bar = .false.
  end
end

!ERROR: Intrinsic operator '.le.' may not be used as a defined operator
use m2, only: operator(.le.) => operator(.ge.)
!ERROR: Intrinsic operator '.not.' may not be used as a defined operator
use m2, only: operator(.not.) => operator(.foo.)
!ERROR: Logical constant '.true.' may not be used as a defined operator
use m2, only: operator(.true.) => operator(.foo.)
end
