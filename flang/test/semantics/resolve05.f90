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

program p
  integer :: p ! this is ok
end
module m
  integer :: m ! this is ok
end
submodule(m) sm
  integer :: sm ! this is ok
end
module m2
  type :: t
  end type
  interface
    subroutine s
      !ERROR: Module 'm2' cannot USE itself.
      use m2, only: t
    end subroutine
  end interface
end module
subroutine s
  !ERROR: 's' is already declared in this scoping unit
  integer :: s
end
function f() result(res)
  integer :: res
  !ERROR: 'f' is already declared in this scoping unit
  !ERROR: The type of 'f' has already been declared
  real :: f
  res = 1
end
