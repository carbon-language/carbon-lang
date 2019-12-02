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

module m
  public i
  integer, private :: j
  !ERROR: The accessibility of 'i' has already been specified as PUBLIC
  private i
  !The accessibility of 'j' has already been specified as PRIVATE
  private j
end

module m2
  interface operator(.foo.)
    module procedure ifoo
  end interface
  public :: operator(.foo.)
  !ERROR: The accessibility of operator '.foo.' has already been specified as PUBLIC
  private :: operator(.foo.)
  interface operator(+)
    module procedure ifoo
  end interface
  public :: operator(+)
  !ERROR: The accessibility of 'operator(+)' has already been specified as PUBLIC
  private :: operator(+) , ifoo
contains
  integer function ifoo(x, y)
    logical, intent(in) :: x, y
  end
end module

module m3
  type t
  end type
  private :: operator(.lt.)
  interface operator(<)
    logical function lt(x, y)
      import t
      type(t), intent(in) :: x, y
    end function
  end interface
  !ERROR: The accessibility of 'operator(<)' has already been specified as PRIVATE
  public :: operator(<)
  interface operator(.gt.)
    logical function gt(x, y)
      import t
      type(t), intent(in) :: x, y
    end function
  end interface
  public :: operator(>)
  !ERROR: The accessibility of 'operator(.gt.)' has already been specified as PUBLIC
  private :: operator(.gt.)
end
