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

module m
  type t1
  contains
    procedure, nopass :: s2
    procedure, nopass :: s3
    generic :: foo => s2
  end type
  type, extends(t1) :: t2
  contains
    procedure, nopass :: s4
    generic :: foo => s3
    generic :: foo => s4
  end type
contains
  subroutine s2(i)
  end
  subroutine s3(r)
  end
  subroutine s4(z)
    complex :: z
  end
end module

!Expect: m.mod
!module m
!  type::t1
!  contains
!    procedure,nopass::s2
!    procedure,nopass::s3
!    generic::foo=>s2
!  end type
!  type,extends(t1)::t2
!  contains
!    procedure,nopass::s4
!    generic::foo=>s2
!    generic::foo=>s3
!    generic::foo=>s4
!  end type
!contains
!  subroutine s2(i)
!    integer(4)::i
!  end
!  subroutine s3(r)
!    real(4)::r
!  end
!  subroutine s4(z)
!    complex(4)::z
!  end
!end
