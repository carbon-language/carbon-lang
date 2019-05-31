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

! See Fortran 2018, clause 17.2

module ieee_features

  type :: ieee_features_type
    private
    integer(kind=1) :: feature = 0
  end type ieee_features_type

  type(ieee_features_type), parameter :: &
    ieee_datatype = ieee_features_type(1), &
    ieee_denormal = ieee_features_type(2), &
    ieee_divide = ieee_features_type(3), &
    ieee_halting = ieee_features_type(4), &
    ieee_inexact_flag = ieee_features_type(5), &
    ieee_inf = ieee_features_type(6), &
    ieee_invalid_flag = ieee_features_type(7), &
    ieee_nan = ieee_features_type(8), &
    ieee_rounding = ieee_features_type(9), &
    ieee_sqrt = ieee_features_type(10), &
    ieee_subnormal = ieee_features_type(11), &
    ieee_underflow_flag = ieee_features_type(12)

end module ieee_features
