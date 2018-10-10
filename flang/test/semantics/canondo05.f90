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
! XXXRUN: ${F18} -fopenmp -funparse-with-symbols %s 2>&1 | ${FileCheck} %s
! CHECK-NOT: do *[1-9]

program P
implicit none
integer OMP_GET_NUM_THREADS, OMP_GET_THREAD_NUM
integer NUMTHRDS, TID
integer N, CSZ, CNUM, I
parameter (N=100)
parameter (CSZ=10) 
real A(N), B(N), C(N)

do 10 I = 1, N
   A(I) = I * 1.0
10 continue

B = A
CNUM = CSZ

!$OMP PARALLEL SHARED(A,B,C,NUMTHRDS,CNUM) PRIVATE(I,TID)
TID = OMP_GET_THREAD_NUM()
if (TID .EQ. 0) then
   NUMTHRDS = OMP_GET_NUM_THREADS()
   print *, "Number of threads =", NUMTHRDS
end if
print *, "Thread", TID, " is starting..."

!$OMP DO SCHEDULE(DYNAMIC,CNUM)
do 20 I = 1, N
   C(I) = A(I) + B(I)
   write (*,100) TID, I, C(I)
20 continue
!$OMP END DO NOWAIT

print *, "Thread", TID, " done."

!$OMP END PARALLEL
100 format(" Thread", I2, ": C(", I3, ")=", F8.2)
end program P
