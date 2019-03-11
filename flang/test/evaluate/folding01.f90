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


! Test intrinsic operation folding

module m
! Check logical intrinsic operation folding
  logical, parameter :: test_not1 = .NOT..false.
  logical, parameter :: test_not2 = .NOT..NOT..true.

  logical, parameter :: test_parentheses1 = .NOT.(.false.)
  logical, parameter :: test_parentheses2 = .NOT..NOT.(.true.)

  logical, parameter :: test_and1 = .true..AND..true.
  logical, parameter :: test_and2 = .NOT.(.false..AND..true.)
  logical, parameter :: test_and3 = .NOT.(.false..AND..false.)
  logical, parameter :: test_and4 = .NOT.(.true..AND..false.)

  logical, parameter :: test_or1 = .true..OR..true.
  logical, parameter :: test_or2 = .false..OR..true.
  logical, parameter :: test_or3 = .NOT.(.false..OR..false.)
  logical, parameter :: test_or4 = .true..OR..false.

  logical, parameter :: test_eqv1 = .false..EQV..false.
  logical, parameter :: test_eqv2 = .true..EQV..true.
  logical, parameter :: test_eqv3 = .NOT.(.false..EQV..true.)
  logical, parameter :: test_eqv4 = .NOT.(.true..EQV..false.)

  logical, parameter :: test_neqv1 = .true..NEQV..false.
  logical, parameter :: test_neqv2 = .false..NEQV..true.
  logical, parameter :: test_neqv3 = .NOT.(.false..NEQV..false.)
  logical, parameter :: test_neqv4 = .NOT.(.true..NEQV..true.)

! Check integer intrinsic operator folding

! Check integer relational intrinsic operation folding
  logical, parameter :: test_le_i1 = 1.LE.2
  logical, parameter :: test_le_i2 = .NOT.(2.LE.1)
  logical, parameter :: test_le_i3 = 2.LE.2
  logical, parameter :: test_le_i4 = -1.LE.2
  logical, parameter :: test_le_i5 = .NOT.(-2.LE.-3)

  logical, parameter :: test_lt_i1 = 1.LT.2
  logical, parameter :: test_lt_i2 = .NOT.(2.LT.1)
  logical, parameter :: test_lt_i3 = .NOT.(2.LT.2)
  logical, parameter :: test_lt_i4 = -1.LT.2
  logical, parameter :: test_lt_i5 = .NOT.(-2.LT.-3)

  logical, parameter :: test_ge_i1 = .NOT.(1.GE.2)
  logical, parameter :: test_ge_i2 =  2.GE.1
  logical, parameter :: test_ge_i3 = 2.GE.2
  logical, parameter :: test_ge_i4 = .NOT.(-1.GE.2)
  logical, parameter :: test_ge_i5 = -2.GE.-3

  logical, parameter :: test_gt_i1 = .NOT.(1.GT.2)
  logical, parameter :: test_gt_i2 =  2.GT.1
  logical, parameter :: test_gt_i3 = .NOT.(2.GT.2)
  logical, parameter :: test_gt_i4 = .NOT.(-1.GT.2)
  logical, parameter :: test_gt_i5 = -2.GT.-3

  logical, parameter :: test_eq_i1 = 2.EQ.2
  logical, parameter :: test_eq_i2 = .NOT.(-2.EQ.2)

  logical, parameter :: test_ne_i1 =.NOT.(2.NE.2)
  logical, parameter :: test_ne_i2 = -2.NE.2

! Check integer intrinsic operation folding
  logical, parameter :: test_unaryminus_i = (-(-1)).EQ.1
  logical, parameter :: test_unaryplus_i = (+1).EQ.1

  logical, parameter :: test_plus_i1 = (1+1).EQ.2
  logical, parameter :: test_plus_i2 = ((-3)+1).EQ.-2

  logical, parameter :: test_minus_i1 = (1-1).EQ.0
  logical, parameter :: test_minus_i2 = (1-(-1)).EQ.2

  logical, parameter :: test_multiply_i1 = (2*2).EQ.4
  logical, parameter :: test_multiply_i2 = (0*1).EQ.0
  logical, parameter :: test_multiply_i3= ((-3)*2).EQ.(-6)

  logical, parameter :: test_divide_i1 = (5/3).EQ.(1)
  logical, parameter :: test_divide_i2 = (6/3).EQ.(2)
  logical, parameter :: test_divide_i3 = ((-7)/2).EQ.(-3)
  logical, parameter :: test_divide_i4 = (0/127).EQ.(0)

  logical, parameter :: test_pow1 = (2**0).EQ.(1)
  logical, parameter :: test_pow2 = (1**100).EQ.(1)
  logical, parameter :: test_pow3 = (2**4).EQ.(16)
  logical, parameter :: test_pow4 = (7**5).EQ.(16807)

end module
