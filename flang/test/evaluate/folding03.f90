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

! Test operation folding edge case (both expected value and messages)
! These tests make assumptions regarding real(4) and integer(4) extrema.

program test
  integer(4), parameter :: i4_neg_max = int(Z'FFFFFFFF')
  integer(4), parameter :: i4_nmax = -2147483647_4 - 1_4 !TODO bug that -2147483648_4 cannot be written ?
  integer(4), parameter :: i4_nmax2 = -2147483648_4
  integer(4), parameter :: i4_pmax1 = 2147483647_4
  integer(4), parameter :: i4_pmax2 = 2147483648_4

print*, "i4_neg_max ", i4_neg_max
print*, "i4_nmax ", i4_nmax
print*, "i4_nmax2 ", i4_nmax2
print*, "i4_pmax1 ", i4_pmax1
print*, "i4_pmax2 ", i4_pmax2

end program

!module integer_tests
  ! These test assume a 
!  integer(4), parameter :: i4_neg_max = int(Z'FFFFFFFF')
!  integer(4), parameter :: i4_pmax = 2147483647_4
!  integer(4), parameter :: i4_nmax = -2147483647_4 - 1_4 !TODO bug that -2147483648_4 cannot be written ?
! ! Or maybe it is a bug that it -2147483648_4 can be reached because it is not
! ! representable in the 16.4 bit model... ??
!
!  ! TODO: Should we test something for 0/0 hw seems to handle this differently (0 or pinf)?
!  ! integer(4), parameter :: i4_nan = (0_4/0_4).EQ.i4_pmax ! Very undefined. Looks like this is zero on power
!
!  !WARN: INTEGER(4) division by zero
!  logical, parameter :: test_pinf = (1_4/0_4).EQ.i4_pmax
!  !WARN: INTEGER(4) division by zero
!  logical, parameter :: test_ninf = (-1_4/0_4).EQ.i4_nmax ! TODO : should it be -2147483648_4 instead of -2147483647_4 ?
!
!  !WARN: INTEGER(4) negation overflowed
!  logical, parameter :: test_overflow_unary_minus1 = (-i4_nmax).EQ.i4_nmax
!  logical, parameter :: test_no_overflow_unary_minus1 = (-i4_pmax).EQ.(i4_nmax+1_4)
!  logical, parameter :: test_no_overflow_unary_plus1 = (+i4_pmax).EQ.i4_pmax
!  logical, parameter :: test_no_overflow_unary_plus2 = (+i4_nmax).EQ.i4_nmax
!
!  !WARN: INTEGER(4) addition overflowed
!  logical, parameter :: test_overflow_add1 = (i4_pmax+1_4).EQ.i4_nmax
!  !WARN: INTEGER(4) addition overflowed
!  logical, parameter :: test_overflow_add2 = (i4_nmax + (-1_4)).EQ.i4_pmax
!  !WARN: INTEGER(4) addition overflowed
!  logical, parameter :: test_overflow_add3 = (i4_pmax + i4_pmax).EQ.(-2_4)
!  !WARN: INTEGER(4) addition overflowed
!  logical, parameter :: test_overflow_add4 = (i4_nmax + i4_nmax).EQ.(0_4)
!  logical, parameter :: test_no_overflow_add1 = (i4_pmax + 0_4).EQ.i4_pmax
!  logical, parameter :: test_no_overflow_add2 = (i4_nmax + (-0_4)).EQ.i4_nmax
!  logical, parameter :: test_no_overflow_add3 = (i4_pmax + i4_nmax).EQ.(-1_4)
!  logical, parameter :: test_no_overflow_add4 = (i4_nmax + i4_pmax).EQ.(-1_4)
!
!  !WARN: INTEGER(4) subtraction overflowed
!  logical, parameter :: test_overflow_sub1 = (i4_nmax - 1_4).EQ.i4_pmax
!  !WARN: INTEGER(4) subtraction overflowed
!  logical, parameter :: test_overflow_sub2 = (i4_pmax - (-1_4)).EQ.i4_nmax
!  !WARN: INTEGER(4) subtraction overflowed
!  logical, parameter :: test_overflow_sub3 = (i4_nmax - i4_pmax).EQ.(1_4)
!  !WARN: INTEGER(4) subtraction overflowed
!  logical, parameter :: test_overflow_sub4 = (i4_pmax - i4_nmax).EQ.(-1_4)
!  logical, parameter :: test_no_overflow_sub1 = (i4_nmax - 0_4).EQ.i4_nmax
!  logical, parameter :: test_no_overflow_sub2 = (i4_pmax - (-0_4)).EQ.i4_pmax
!  logical, parameter :: test_no_overflow_sub3 = (i4_nmax - i4_nmax).EQ.0_4
!  logical, parameter :: test_no_overflow_sub4 = (i4_pmax - i4_pmax).EQ.0_4
!
!
!  !WARN: INTEGER(4) multiplication overflowed
!  logical, parameter :: test_overflow_mult1 = (i4_pmax*2_4).EQ.(-2_4)
!  !WARN: INTEGER(4) multiplication overflowed
!  logical, parameter :: test_overflow_mult2 = (i4_nmax*2_4).EQ.(0_4)
!  !WARN: INTEGER(4) multiplication overflowed
!  logical, parameter :: test_overflow_mult3 = (i4_nmax*i4_nmax).EQ.(0_4)
!  !WARN: INTEGER(4) multiplication overflowed
!  logical, parameter :: test_overflow_mult4 = (i4_pmax*i4_pmax).EQ.(1_4)

!end module

!module real_tests
!  ! Test real(4) operation folding on edge cases (inf and NaN)
!
!  real(4), parameter :: r4_pmax = 3.4028235E38
!  real(4), parameter :: r4_nmax = -3.4028235E38
!  !WARN: invalid argument on division
!  real(4), parameter :: r4_nan = (0._4/0._4)
!  !WARN: division by zero on division
!  real(4), parameter :: r4_pinf = (1._4/0._4)
!  !WARN: division by zero on division
!  real(4), parameter :: r4_ninf = (-1._4/0._4)
!
!
!  ! No warnings expected
!  logical, parameter :: test_r4_negation1 = (-r4_pmax).EQ.r4_nmax
!  logical, parameter :: test_r4_negation2 = (-r4_nmax).EQ.r4_pmax
!  logical, parameter :: test_r4_negation3 = (-r4_pinf).EQ.r4_ninf
!  logical, parameter :: test_r4_negation4 = (-r4_ninf).EQ.r4_pinf
!  logical, parameter :: test_r4_plus1 = (+r4_pmax).EQ.r4_pmax
!  logical, parameter :: test_r4_plus2 = (+r4_nmax).EQ.r4_nmax
!  logical, parameter :: test_r4_plus3 = (+r4_pinf).EQ.r4_pinf
!  logical, parameter :: test_r4_plus4 = (+r4_ninf).EQ.r4_ninf
!  ! NaN propagation , no warnings expected (quiet)
!  real(4), parameter :: r_r4_nan_minus = (-r4_nan)
!  real(4), parameter :: r_r4_nan_plus = (+r4_nan)
!
!  !WARN: overflow on addition
!  logical, parameter :: test_inf_r4_add9 = (r4_pmax + r4_pmax).eq.(r4_pinf)
!  !WARN: overflow on addition
!  logical, parameter :: test_inf_r4_add10 = (r4_nmax + r4_nmax).eq.(r4_ninf)
!  !WARN: overflow on subtraction
!  logical, parameter :: test_inf_r4_sub9 = (r4_pmax - r4_nmax).eq.(r4_pinf)
!  !WARN: overflow on subtraction
!  logical, parameter :: test_inf_r4_sub10 = (r4_nmax - r4_pmax).eq.(r4_ninf)
!
!  ! No warnings expected below (inf propagation).
!  logical, parameter :: test_inf_r4_add1 = (r4_pinf + r4_pinf).EQ.(r4_pinf)
!  logical, parameter :: test_inf_r4_add2 = (r4_ninf + r4_ninf).EQ.(r4_ninf)
!  logical, parameter :: test_inf_r4_add3 = (r4_pinf + r4_nmax).EQ.(r4_pinf)
!  logical, parameter :: test_inf_r4_add4 = (r4_pinf + r4_pmax).EQ.(r4_pinf)
!  logical, parameter :: test_inf_r4_add5 = (r4_ninf + r4_pmax).EQ.(r4_ninf)
!  logical, parameter :: test_inf_r4_add6 = (r4_ninf + r4_nmax).EQ.(r4_ninf)
!  logical, parameter :: test_inf_r4_add7 = (r4_ninf + 0._4).EQ.(r4_ninf)
!  logical, parameter :: test_inf_r4_add8 = (r4_pinf + 0._4).EQ.(r4_pinf)
!
!  !WARN: invalid argument on subtraction
!  real(4), parameter :: r_nan_r4_sub1 = r4_pinf - r4_pinf
!  !WARN: invalid argument on subtraction
!  real(4), parameter :: r_nan_r4_sub2 = r4_ninf - r4_ninf
!  !WARN: invalid argument on addition
!  real(4), parameter :: r_nan_r4_add1 = r4_ninf + r4_pinf
!  !WARN: invalid argument on addition
!  real(4), parameter :: r_nan_r4_add2 = r4_pinf + r4_ninf
!
!  ! No warnings expected here (quite NaN propagation)
!  real(4), parameter :: r_nan_r4_sub3 = 0._4 - r4_nan
!  real(4), parameter :: r_nan_r4_sub4 = r4_nan - r4_pmax
!  real(4), parameter :: r_nan_r4_sub5 = r4_nan - r4_nmax
!  real(4), parameter :: r_nan_r4_sub6 = r4_nan - r4_nan
!  real(4), parameter :: r_nan_r4_add3 = 0._4 + r4_nan
!  real(4), parameter :: r_nan_r4_add4 = r4_nan + r4_pmax
!  real(4), parameter :: r_nan_r4_add5 = r4_nmax + r4_nan
!  real(4), parameter :: r_nan_r4_add6 = r4_nan + r4_nan
!
!  !WARN: overflow on multiplication
!  logical, parameter :: test_inf_r4_mult1 = (1.5_4*r4_pmax).eq.(r4_pinf)
!  !WARN: overflow on multiplication
!  logical, parameter :: test_inf_r4_mult2 = (1.5_4*r4_nmax).eq.(r4_ninf)
!  !WARN: overflow on division
!  logical, parameter :: test_inf_r4_div1 = (r4_nmax/(-0.5_4)).eq.(r4_pinf)
!  !WARN: overflow on division
!  logical, parameter :: test_inf_r4_div2 = (r4_pmax/(-0.5_4)).eq.(r4_ninf)
!
!  ! No warnings expected below (inf propagation).
!  logical, parameter :: test_inf_r4_mult3 = (r4_pinf*r4_pinf).EQ.(r4_pinf)
!  logical, parameter :: test_inf_r4_mult4 = (r4_ninf*r4_ninf).EQ.(r4_pinf)
!  logical, parameter :: test_inf_r4_mult5 = (r4_pinf*0.1_4).EQ.(r4_pinf)
!  logical, parameter :: test_inf_r4_mult6 = (r4_ninf*r4_nmax).EQ.(r4_pinf)
!  logical, parameter :: test_inf_r4_div3 = (r4_pinf/0.).EQ.(r4_pinf)
!  logical, parameter :: test_inf_r4_div4 = (r4_ninf/0.).EQ.(r4_ninf)
!  logical, parameter :: test_inf_r4_div5 = (0./r4_pinf).EQ.(0.)
!  logical, parameter :: test_inf_r4_div6 = (0./r4_ninf).EQ.(0.)
!  logical, parameter :: test_inf_r4_div7 = (r4_pinf/r4_pmax).EQ.(r4_pinf)
!  logical, parameter :: test_inf_r4_div8 = (r4_pinf/r4_nmax).EQ.(r4_ninf)
!  logical, parameter :: test_inf_r4_div9 = (r4_nmax/r4_pinf).EQ.(0.)
!  logical, parameter :: test_inf_r4_div10 = (r4_nmax/r4_ninf).EQ.(0.)
!
!  !WARN: invalid argument on division
!  real(4), parameter :: r_nan_r4_div1 = 0._4/0._4
!  !WARN: invalid argument on division
!  real(4), parameter :: r_nan_r4_div2 = r4_ninf/r4_ninf
!  !WARN: invalid argument on division
!  real(4), parameter :: r_nan_r4_div3 = r4_ninf/r4_pinf
!  !WARN: invalid argument on division
!  real(4), parameter :: r_nan_r4_div4 = r4_pinf/r4_ninf
!  !WARN: invalid argument on division
!  real(4), parameter :: r_nan_r4_div5 = r4_pinf/r4_pinf
!  !WARN: invalid argument on multiplication
!  real(4), parameter :: r_nan_r4_mult1 = r4_pinf*0._4
!  !WARN: invalid argument on multiplication
!  real(4), parameter :: r_nan_r4_mult2 = 0._4*r4_ninf
!
!  ! No warnings expected here (quite NaN propagation)
!  real(4), parameter :: r_nan_r4_div6 = 0._4/r4_nan
!  real(4), parameter :: r_nan_r4_div7 = r4_nan/r4_nan
!  real(4), parameter :: r_nan_r4_div8 = r4_nan/0._4
!  real(4), parameter :: r_nan_r4_div9 = r4_nan/1._4
!  real(4), parameter :: r_nan_r4_mult3 = r4_nan*1._4
!  real(4), parameter :: r_nan_r4_mult4 = r4_nan*r4_nan
!  real(4), parameter :: r_nan_r4_mult5 = 0._4*r4_nan
!
!  ! TODO: ** operator folding
!  !  logical, parameter :: test_inf_r4_exp1 = (r4_pmax**2._4).EQ.(r4_pinf)
!
!  ! Relational operator edge cases (No warnings expected?)
!  logical, parameter :: test_inf_r4_eq1 = r4_pinf.EQ.r4_pinf
!  logical, parameter :: test_inf_r4_eq2 = r4_ninf.EQ.r4_ninf
!  logical, parameter :: test_inf_r4_eq3 = .NOT.(r4_pinf.EQ.r4_ninf)
!  logical, parameter :: test_inf_r4_eq4 = .NOT.(r4_pinf.EQ.r4_pmax)
!
!  logical, parameter :: test_inf_r4_ne1 = .NOT.(r4_pinf.NE.r4_pinf)
!  logical, parameter :: test_inf_r4_ne2 = .NOT.(r4_ninf.NE.r4_ninf)
!  logical, parameter :: test_inf_r4_ne3 = r4_pinf.NE.r4_ninf
!  logical, parameter :: test_inf_r4_ne4 = r4_pinf.NE.r4_pmax
!
!  logical, parameter :: test_inf_r4_gt1 = .NOT.(r4_pinf.GT.r4_pinf)
!  logical, parameter :: test_inf_r4_gt2 = .NOT.(r4_ninf.GT.r4_ninf)
!  logical, parameter :: test_inf_r4_gt3 = r4_pinf.GT.r4_ninf
!  logical, parameter :: test_inf_r4_gt4 = r4_pinf.GT.r4_pmax
!
!  logical, parameter :: test_inf_r4_lt1 = .NOT.(r4_pinf.LT.r4_pinf)
!  logical, parameter :: test_inf_r4_lt2 = .NOT.(r4_ninf.LT.r4_ninf)
!  logical, parameter :: test_inf_r4_lt3 = r4_ninf.LT.r4_pinf
!  logical, parameter :: test_inf_r4_lt4 = r4_pmax.LT.r4_pinf
!
!  logical, parameter :: test_inf_r4_ge1 = r4_pinf.GE.r4_pinf
!  logical, parameter :: test_inf_r4_ge2 = r4_ninf.GE.r4_ninf
!  logical, parameter :: test_inf_r4_ge3 = .NOT.(r4_ninf.GE.r4_pinf)
!  logical, parameter :: test_inf_r4_ge4 = .NOT.(r4_pmax.GE.r4_pinf)
!
!  logical, parameter :: test_inf_r4_le1 = r4_pinf.LE.r4_pinf
!  logical, parameter :: test_inf_r4_le2 = r4_ninf.LE.r4_ninf
!  logical, parameter :: test_inf_r4_le3 = .NOT.(r4_pinf.LE.r4_ninf)
!  logical, parameter :: test_inf_r4_le4 = .NOT.(r4_pinf.LE.r4_pmax)
!
!  ! Invalid relational argument ? Defined behavior to return false ?
!  logical, parameter :: test_nan_r4_eq1 = .NOT.(r4_nan.EQ.r4_nan)
!  logical, parameter :: test_nan_r4_ne1 = .NOT.(r4_nan.NE.r4_nan)
!
!  !real(4), parameter :: acos_nan_r10 = acos(r4_pinf)
!
!  ! logical, parameter :: test_add_overflow = (2147483647_4 + 2_4).EQ.(-2147483647_4)
!  ! ! logical, parameter :: test_add_overflow2 = .NOT.(2147483647_4 + 2_4).EQ.(+1/0.)
!  ! real, parameter :: test_error = acos(2.) ! This is NAN. Can we test that ?? ISNAN non standard intrinsic?
!  ! logical, parameter :: test_host_overflow =  exp(256._4).EQ.(1./0.)
!  ! logical, parameter :: test_not_host_overflow =  exp(256._8).NE.(1./0.)
!  
!  ! TODO: conversions
!  ! integer(4), parameter :: r4_nan_to_i4 = int(r4_nan, 4)
!  ! integer(4), parameter :: r4_pinf_to_i4 = int(r4_pinf, 4)
!  ! integer(4), parameter :: r4_ninf_to_i4 = int(r4_ninf, 4)
!  !
!  ! integer(16), parameter :: r4_nan_to_i16 = int(r4_nan, 16)
!  ! integer(16), parameter :: r4_pinf_to_i16 = int(r4_pinf, 16)
!  ! integer(16), parameter :: r4_ninf_to_i16 = int(r4_ninf, 16)
!  !  ! TODO: That seems wrong to me. These should stay what they are
!  ! real, parameter :: test_conversion_from_nan_to_real = real(test_error, 8)
!  ! real, parameter :: test_conversion_from_pinf_to_real = real(1._4/0._4, 8)
!  ! real, parameter :: test_conversion_from_ninf_to_real = real(-1._4/0._4, 8)
!  ! !real, parameter :: test_error2 = 0./0.
!  ! !real, parameter :: test_error3 = acos(2.)
!  ! !integer, parameter :: test_error4 = int(-1./0.)
!  ! !real(kind=test_error4) y
!  ! !real(kind=(2147483647_4 - (2147483647_4 + 2_4) + 4_4)) x
!
!  ! TODO: character intrinsic folding
!  !
!end module

! TODO complex tests ?

! Logical operation (with logical arguments) cannot overflow or be invalid.
! CHARACTER folding operations may cause host memory exhaustion if the
! string are very large. This will cause a fatal error for the program
! doing folding (e.g. f18), so there is nothing very interesting to test here.
