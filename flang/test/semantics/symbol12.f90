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

! Verify that SAVE attribute is propagated by EQUIVALENCE

!DEF: /s1 (Subroutine) Subprogram
subroutine s1
 !DEF: /s1/a SAVE ObjectEntity REAL(4)
 !DEF: /s1/b SAVE ObjectEntity REAL(4)
 !DEF: /s1/c SAVE ObjectEntity REAL(4)
 !DEF: /s1/d SAVE ObjectEntity REAL(4)
 real a, b, c, d
 !REF: /s1/d
 save :: d
 !REF: /s1/a
 !REF: /s1/b
 equivalence(a, b)
 !REF: /s1/b
 !REF: /s1/c
 equivalence(b, c)
 !REF: /s1/c
 !REF: /s1/d
 equivalence(c, d)
 !DEF: /s1/e ObjectEntity INTEGER(4)
 !DEF: /s1/f ObjectEntity INTEGER(4)
 equivalence(e, f)
 !REF: /s1/e
 !REF: /s1/f
 integer e, f
end subroutine
