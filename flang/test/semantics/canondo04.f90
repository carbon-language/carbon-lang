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

! RUN: ${F18} -funparse-with-symbols %s 2>&1 | ${FileCheck} %s
! CHECK: end do

program main
  do 1 j1=1,2
    do 1 j2=1,2
      if (j1 == j2) then
        do 2 j3=1,2
          print *, j1, j2, j3
          do 2 j4=1,2
            print *, j3, j4
2         end do
      else
        do 3 j3=3,4
          print *, j1, j2, j3
          do 3 j4=3,4
            print *, j3, j4
3         end do
      end if
    print *, j1, j2
1   continue
  do 4 j1=3,4 ! adjacent non-block DO loops
4   print *, j1
  do 5 j1=5,6 ! non-block DO loop at end of execution part
5   print *, j1
end
