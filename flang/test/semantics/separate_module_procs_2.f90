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

module m1
  interface ma
  end interface
end module

submodule (m1) ma_submodule
  contains
  !ERROR: 'ma_create_new_fun' was not declared a separate module procedure
  module function ma_create_new_fun() result(this)
    integer :: this
    print *, "Hello"
  end function
end submodule

module m2
  interface mb
  end interface
end module

submodule (m2) mb_submodule
  contains
  !ERROR: 'mb_create_new_sub' was not declared a separate module procedure
  module SUBROUTINE  mb_create_new_sub() 
    integer :: this
    print *, "Hello"
  end SUBROUTINE mb_create_new_sub
end submodule
