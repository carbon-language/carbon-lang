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

program main
  intrinsic :: cos ! a specific & generic intrinsic name
  intrinsic :: alog10 ! a specific intrinsic name, not generic
  intrinsic :: null ! a weird special case
  !ERROR: 'haltandcatchfire' is not a known intrinsic procedure
  intrinsic :: haltandcatchfire
  procedure(sin), pointer :: p
  p => alog ! valid use of an unrestricted specific intrinsic
  p => alog10 ! ditto, but already declared intrinsic
  p => cos ! ditto, but also generic
  p => tan ! a generic & an unrestricted specific, not already declared
  !TODO ERROR: a restricted specific, to be caught in ass't semantics
  p => amin1
end program main
