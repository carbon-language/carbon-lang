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

! Check modfile generation for generic interfaces
module m
  interface foo
    real function s1(x,y)
      real x,y
    end function
    complex function s2(x,y)
      complex x,y
    end function
  end interface
  generic :: operator ( + ) => s1, s2
  interface bar
    procedure :: s1
    procedure :: s2
    procedure :: s3
    procedure :: s4
  end interface
  interface operator( .bar.)
    procedure :: s1
    procedure :: s2
    procedure :: s3
    procedure :: s4
  end interface
contains
  logical function s3(x,y)
    logical x,y
  end function
  integer function s4(x,y)
    integer x,y
  end function
end
!Expect: m.mod
!module m
! interface foo
!  procedure::s1
!  procedure::s2
! end interface
! interface
!  function s1(x,y)
!   real(4)::x
!   real(4)::y
!   real(4)::s1
!  end
! end interface
! interface
!  function s2(x,y)
!   complex(4)::x
!   complex(4)::y
!   complex(4)::s2
!  end
! end interface
! interface operator(+)
!  procedure::s1
!  procedure::s2
! end interface
! interface bar
!  procedure::s1
!  procedure::s2
!  procedure::s3
!  procedure::s4
! end interface
! interface operator(.bar.)
!  procedure::s1
!  procedure::s2
!  procedure::s3
!  procedure::s4
! end interface
!contains
! function s3(x,y)
!  logical(4)::x
!  logical(4)::y
!  logical(4)::s3
! end
! function s4(x,y)
!  integer(4)::x
!  integer(4)::y
!  integer(4)::s4
! end
!end

module m2
  interface foo
    procedure foo
  end interface
contains
  complex function foo()
    foo = 1.0
  end
end
!Expect: m2.mod
!module m2
! interface foo
!  procedure::foo
! end interface
!contains
! function foo()
!  complex(4)::foo
! end
!end

module m2b
  type :: foo
    real :: x
  end type
  interface foo
  end interface
  private :: bar
  interface bar
  end interface
end
!Expect: m2b.mod
!module m2b
! interface foo
! end interface
! type::foo
!  real(4)::x
! end type
! interface bar
! end interface
! private::bar
!end

! Test interface nested inside another interface
module m3
  interface g
    subroutine s1(f)
      interface
        real function f(x)
          interface
            subroutine x()
            end subroutine
          end interface
        end function
      end interface
    end subroutine
  end interface
end
!Expect: m3.mod
!module m3
! interface g
!  procedure::s1
! end interface
! interface
!  subroutine s1(f)
!   interface
!    function f(x)
!     interface
!      subroutine x()
!      end
!     end interface
!     real(4)::f
!    end
!   end interface
!  end
! end interface
!end

module m4
  interface foo
    integer function foo()
    end function
    integer function f(x)
    end function
  end interface
end
subroutine s4
  use m4
  i = foo()
end
!Expect: m4.mod
!module m4
! interface foo
!  procedure::foo
!  procedure::f
! end interface
! interface
!  function foo()
!   integer(4)::foo
!  end
! end interface
! interface
!  function f(x)
!   real(4)::x
!   integer(4)::f
!  end
! end interface
!end

! Compile contents of m4.mod and verify it gets the same thing again.
module m5
 interface foo
  procedure::foo
  procedure::f
 end interface
 interface
  function foo()
   integer(4)::foo
  end
 end interface
 interface
  function f(x)
   integer(4)::f
   real(4)::x
  end
 end interface
end
!Expect: m5.mod
!module m5
! interface foo
!  procedure::foo
!  procedure::f
! end interface
! interface
!  function foo()
!   integer(4)::foo
!  end
! end interface
! interface
!  function f(x)
!   real(4)::x
!   integer(4)::f
!  end
! end interface
!end
