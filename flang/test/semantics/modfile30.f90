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

! Verify miscellaneous bugs

! The function result must be declared after the dummy arguments
module m1
contains
  function f1(x) result(y)
    integer :: x(:)
    integer :: y(size(x))
  end
  function f2(x)
    integer :: x(:)
    integer :: f2(size(x))
  end
end

!Expect: m1.mod
!module m1
!contains
! function f1(x) result(y)
!  integer(4)::x(:)
!  integer(4)::y(1_8:1_8*size(x,dim=1))
! end
! function f2(x)
!  integer(4)::x(:)
!  integer(4)::f2(1_8:1_8*size(x,dim=1))
! end
!end
