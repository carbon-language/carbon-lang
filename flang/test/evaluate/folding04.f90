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

! Test intrinsic function folding edge case (both expected value and messages)
! These tests make assumptions regarding real(4) extrema.

!end program

#define TEST_ISNAN(v) logical, parameter :: test_##v =.NOT.(v.EQ.v)


module real_tests
  ! Test real(4) intrinsic folding on edge cases (inf and NaN)

  real(4), parameter :: r4_pmax = 3.4028235E38
  real(4), parameter :: r4_nmax = -3.4028235E38
  !WARN: invalid argument on division
  real(4), parameter :: r4_nan = 0._4/0._4
  !WARN: division by zero on division
  real(4), parameter :: r4_pinf = 1._4/0._4
  !WARN: division by zero on division
  real(4), parameter :: r4_ninf = -1._4/0._4

  !WARN: invalid argument on folding function with host runtime
  real(4), parameter :: nan_r4_acos1 = acos(1.1)
  TEST_ISNAN(nan_r4_acos1)
  !WARN: invalid argument on folding function with host runtime
  real(4), parameter :: nan_r4_acos2 = acos(r4_pmax)
  TEST_ISNAN(nan_r4_acos2)
  !WARN: invalid argument on folding function with host runtime
  real(4), parameter :: nan_r4_acos3 = acos(r4_nmax)
  TEST_ISNAN(nan_r4_acos3)
  !WARN: invalid argument on folding function with host runtime
  real(4), parameter :: nan_r4_acos4 = acos(r4_ninf)
  TEST_ISNAN(nan_r4_acos4)
  !WARN: invalid argument on folding function with host runtime
  real(4), parameter :: nan_r4_acos5 = acos(r4_pinf)
  TEST_ISNAN(nan_r4_acos5)
  ! No warnings expected for NaN propagation (quiet)
  real(4), parameter :: nan_r4_acos6 = acos(r4_nan)
  TEST_ISNAN(nan_r4_acos6)

  !WARN: overflow on folding function with host runtime
  logical, parameter :: test_exp_overflow = exp(256._4).EQ.r4_pinf
end module

module parentheses
  ! Test parentheses in folding (they are kept around constants to keep the
  ! distinction between variable and expressions and require special care).
  real(4), parameter :: x_nop = 0.1_4
  real(4), parameter :: x_p = (x_nop)
  logical, parameter :: test_parentheses1 = acos(x_p).EQ.acos(x_nop)
end module
