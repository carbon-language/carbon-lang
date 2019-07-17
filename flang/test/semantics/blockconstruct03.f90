! Copyright (c) 2019, ARM Ltd.  All rights reserved.
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
! Tests implemented fro this standard
! 11.1.4 - 4 It is permissible to branch to and en-block-stm only withinh its
!            Block Construct
! C1109
! RUN: ${F18} -fdebug-semantics   %s 2>&1 | ${FileCheck} %s
! CHECK: label '20' is not in scope
! CHECK: should be
subroutine s1
  block
    goto (10) 1
10  end block

  block
20  end block
end

subroutine s2
  block
    goto (20) 1
10  end block

  block
20  end block
end

subroutine s3
  block
    block
      goto (10) 1
10  end block
20  end block
end

subroutine s4
  block
    block
      goto (20) 1
10  end block
20  end block
end

subroutine s5_c1109
  b1:block
  !ERROR: BLOCK construct name mismatch
  end block b2
end

