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

module m
  type :: t
    procedure(a), pointer, pass :: c
    procedure(a), pointer, pass(x) :: d
  contains
    procedure, pass(y) :: a, b
  end type
contains
  subroutine a(x, y)
    class(t) :: x, y
  end
  subroutine b(y)
    class(t) :: y
  end
end module

!Expect: m.mod
!module m
!  type::t
!    procedure(a),pass,pointer::c
!    procedure(a),pass(x),pointer::d
!  contains
!    procedure,pass(y)::a
!    procedure,pass(y)::b
!  end type
!contains
!  subroutine a(x,y)
!    class(t)::x
!    class(t)::y
!  end
!  subroutine b(y)
!    class(t)::y
!  end
!end
