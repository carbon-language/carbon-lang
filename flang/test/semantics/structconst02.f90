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

! Error tests for structure constructors: per-component type
! (in)compatibility.

module module1
  interface
    real function realfunc(x)
      real, value :: x
    end function realfunc
  end interface
  type :: scalar(ik,rk,zk,ck,lk,len)
    integer, kind :: ik = 4, rk = 4, zk = 4, ck = 1, lk = 1
    integer, len :: len = 1
    integer(kind=ik) :: ix = 0
    real(kind=rk) :: rx = 0.
    complex(kind=zk) :: zx = (0.,0.)
    character(kind=ck,len=len) :: cx = ' '
    logical(kind=lk) :: lx = .false.
    real(kind=rk), pointer :: rp = NULL()
    procedure(realfunc), pointer :: rfp1 => NULL()
    procedure(real), pointer :: rfp2 => NULL()
  end type scalar
 contains
  subroutine scalararg(x)
    type(scalar), intent(in) :: x
  end subroutine scalararg
  subroutine errors
    call scalararg(scalar(4)(ix=1,rx=2.,zx=(3.,4.),cx='a',lx=.true.))
    call scalararg(scalar(4)(1,2.,(3.,4.),'a',.true.))
!    call scalararg(scalar(4)(ix=5.,rx=6,zx=(7._8,8._2),cx=4_'b',lx=.true._4))
!    call scalararg(scalar(4)(5.,6,(7._8,8._2),4_'b',.true._4))
    call scalararg(scalar(4)(ix=5.,rx=6,zx=(7._8,8._2),cx=4_'b',lx=.true.))
    call scalararg(scalar(4)(5.,6,(7._8,8._2),4_'b',.true.))
    call scalararg(scalar(4)(ix='a'))
    call scalararg(scalar(4)(ix=.false.))
    call scalararg(scalar(4)(ix=[1]))
    !TODO more!
  end subroutine errors
end module module1
