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

! Test that associations constructs can be correctly combined. The intrinsic
! functions are not what is tested here, they are only use to reveal the types
! of local variables.

  implicit none
  real res
  complex zres
  integer ires
  class(*), allocatable :: a, b
  select type(a)
    type is (integer)
      select type(b)
        type is (integer)
          ires = selected_int_kind(b)
          ires = selected_int_kind(a)
      end select
    type is (real)
     res = acos(a)
     !ERROR: Actual argument for 'x=' has bad type 'CLASS(*)'
     res = acos(b)
  end select

  select type(c => a)
    type is (real)
     res = acos(c)
    class default
     !ERROR: Actual argument for 'x=' has bad type 'CLASS(*)'
     res = acos(c)
  end select
  select type(a)
    type is (integer)
     !ERROR: Actual argument for 'x=' has bad type 'Integer(4)'
     res = acos(a)
  end select

  select type(b)
    type is (integer)
      associate(y=>1.0, x=>1, z=>(1.0,2.3))
        ires = selected_int_kind(x)
        select type(a)
          type is (real)
            res = acos(a)
            res = acos(y)
            !ERROR: Actual argument for 'x=' has bad type 'Integer(4)'
            res = acos(b)
          type is (integer)
            ires = selected_int_kind(b)
            zres = acos(z)
           !ERROR: Actual argument for 'x=' has bad type 'Integer(4)'
           res = acos(a)
        end select
      end associate
      ires = selected_int_kind(b)
      !ERROR: No explicit type declared for 'c'
      ires = selected_int_kind(c)
      !ERROR: Actual argument for 'x=' has bad type 'CLASS(*)'
      res = acos(a)
    class default
      !ERROR: Actual argument for 'r=' has bad type 'CLASS(*)'
      ires = selected_int_kind(b)
  end select
  !ERROR: Actual argument for 'r=' has bad type 'CLASS(*)'
  ires = selected_int_kind(a)
  !ERROR: Actual argument for 'x=' has bad type 'CLASS(*)'
  res = acos(b)
end
