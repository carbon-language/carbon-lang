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

!DEF: /m1 Module
module m1
  !DEF: /m1/ma PUBLIC (Function) Generic
  interface ma
    !DEF: /m1/ma_create_fun MODULE, PUBLIC (Function) Subprogram INTEGER(4)
    !DEF: /m1/ma_create_fun/this ObjectEntity INTEGER(4)
    module function ma_create_fun( ) result(this)
      !REF: /m1/ma_create_fun/this
      integer this
    end function
  end interface
end module

!REF: /m1
!DEF: /m1/ma_submodule Module
submodule (m1) ma_submodule
  contains
  !DEF: /m1/ma_submodule/ma_create_fun MODULE, PUBLIC (Function) Subprogram INTEGER(4)
  !DEF: /m1/ma_submodule/ma_create_fun/this ObjectEntity INTEGER(4)
  module function ma_create_fun() result(this)
    !REF: /m1/ma_submodule/ma_create_fun/this
    integer this
    print *, "Hello"
  end function
end submodule

!DEF: /m2 Module
module m2
  !DEF: /m2/mb PUBLIC (Subroutine) Generic
  interface mb
    !DEF: /m2/mb_create_sub MODULE, PUBLIC (Subroutine) Subprogram
    module subroutine  mb_create_sub
    end subroutine mb_create_sub
  end interface
end module

!REF: /m2
!DEF: /m2/mb_submodule Module
submodule (m2) mb_submodule
  contains
  !DEF: /m2/mb_submodule/mb_create_sub MODULE, PUBLIC (Subroutine) Subprogram
  module subroutine  mb_create_sub
    !DEF: /m2/mb_submodule/mb_create_sub/this ObjectEntity INTEGER(4)
    integer this
    print *, "Hello"
  end subroutine mb_create_sub
end submodule
