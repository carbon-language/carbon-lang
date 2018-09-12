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

! Check modfile generation for external declarations
module m
  real, external :: a
  logical b
  external c
  complex c
  external b, d
  procedure() :: e
  procedure(real) :: f
  procedure(s) :: g
  type t
    procedure(), pointer, nopass :: e
    procedure(real), nopass, pointer :: f
    procedure(s), private, pointer :: g
  end type
contains
  subroutine s(x)
    class(t) :: x
  end
end

!Expect: m.mod
!module m
!  procedure(real(4))::a
!  procedure(logical(4))::b
!  procedure(complex(4))::c
!  procedure()::d
!  procedure()::e
!  procedure(real(4))::f
!  procedure(s)::g
!  type::t
!    procedure(),nopass,pointer::e
!    procedure(real(4)),nopass,pointer::f
!    procedure(s),pointer,private::g
!  end type
!contains
!  subroutine s(x)
!    class(t)::x
!  end
!end
