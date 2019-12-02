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
  interface foo
    subroutine s1(x)
      real x
    end
    !ERROR: 's2' is not a module procedure
    module procedure s2
    !ERROR: Procedure 's3' not found
    procedure s3
    !ERROR: Procedure 's1' is already specified in generic 'foo'
    procedure s1
  end interface
  interface
    subroutine s4(x,y)
      real x,y
    end subroutine
    subroutine s2(x,y)
      complex x,y
    end subroutine
  end interface
  generic :: bar => s4
  generic :: bar => s2
  !ERROR: Procedure 's4' is already specified in generic 'bar'
  generic :: bar => s4

  generic :: operator(.foo.)=> s4
  generic :: operator(.foo.)=> s2
  !ERROR: Procedure 's4' is already specified in generic operator '.foo.'
  generic :: operator(.foo.)=> s4
end module

module m2
  interface
    integer function f(x, y)
      logical, intent(in) :: x, y
    end function
  end interface
  generic :: operator(+)=> f
  !ERROR: Procedure 'f' is already specified in generic 'operator(+)'
  generic :: operator(+)=> f
end

module m3
  interface operator(.ge.)
    procedure f
  end interface
  interface operator(>=)
    !ERROR: Procedure 'f' is already specified in generic 'operator(.ge.)'
    procedure f
  end interface
  generic :: operator(>) => f
  !ERROR: Procedure 'f' is already specified in generic 'operator(>)'
  generic :: operator(.gt.) => f
contains
  logical function f(x, y) result(result)
    logical, intent(in) :: x, y
    result = .true.
  end
end
