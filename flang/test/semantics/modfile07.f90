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
! generic::foo=>s1,s2
! interface
!  function s1(x,y)
!   real(4)::s1
!   real(4)::x
!   real(4)::y
!  end
! end interface
! interface
!  function s2(x,y)
!   complex(4)::s2
!   complex(4)::x
!   complex(4)::y
!  end
! end interface
! generic::operator(+)=>s1,s2
! generic::bar=>s1,s2,s3,s4
! generic::operator(.bar.)=>s1,s2,s3,s4
!contains
! function s3(x,y)
!  logical(4)::s3
!  logical(4)::x
!  logical(4)::y
! end
! function s4(x,y)
!  integer(4)::s4
!  integer(4)::x
!  integer(4)::y
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
! generic::foo=>foo
!contains
! function foo()
!  complex(4)::foo
! end
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
! generic::g=>s1
! interface
!  subroutine s1(f)
!   interface
!    function f(x)
!     real(4)::f
!     interface
!      subroutine x()
!      end
!     end interface
!    end
!   end interface
!  end
! end interface
!end
