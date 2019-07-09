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

! OPTIONS: -fopenmp

! Check OpenMP clause validity for the following directives:
!
!    2.5 PARALLEL construct
!    2.7.1 Loop construct
!    ...

! TODO: all the internal errors

  integer :: b = 128
  integer :: c = 32
  integer, parameter :: num = 16
  N = 1024

! 2.5 parallel-clause -> if-clause |
!                        num-threads-clause |
!                        default-clause |
!                        private-clause |
!                        firstprivate-clause |
!                        shared-clause |
!                        copyin-clause |
!                        reduction-clause |
!                        proc-bind-clause

  !$omp parallel
  do i = 1, N
     a = 3.14
  enddo
  !$omp end parallel

  !ERROR: SCHEDULE clause is not allowed on the PARALLEL directive
  !$omp parallel schedule(static)
  do i = 1, N
     a = 3.14
  enddo
  !$omp end parallel

  !ERROR: COLLAPSE clause is not allowed on the PARALLEL directive
  !$omp parallel collapse(2)
  do i = 1, N
     a = 3.14
  enddo
  !$omp end parallel

  a = 1.0
  !$omp parallel firstprivate(a)
  do i = 1, N
     a = 3.14
  enddo
  !$omp end parallel

  !ERROR: LASTPRIVATE clause is not allowed on the PARALLEL directive
  !ERROR: NUM_TASKS clause is not allowed on the PARALLEL directive
  !ERROR: INBRANCH clause is not allowed on the PARALLEL directive
  !$omp parallel lastprivate(a) NUM_TASKS(4) inbranch
  do i = 1, N
     a = 3.14
  enddo
  !$omp end parallel

  !ERROR: At most one NUM_THREADS clause can appear on the PARALLEL directive
  !$omp parallel num_threads(2) num_threads(4)
  do i = 1, N
     a = 3.14
  enddo
  !$omp end parallel

  !ERROR: The parameter of the NUM_THREADS clause must be a positive integer expression
  !$omp parallel num_threads(1-4)
  do i = 1, N
     a = 3.14
  enddo
  !$omp end parallel

  !$omp parallel num_threads(num-10)
  do i = 1, N
     a = 3.14
  enddo
  !$omp end parallel

  !$omp parallel num_threads(b+1)
  do i = 1, N
     a = 3.14
  enddo
  !$omp end parallel

! 2.7.1  do-clause -> private-clause |
!                     firstprivate-clause |
!                     lastprivate-clause |
!                     linear-clause |
!                     reduction-clause |
!                     schedule-clause |
!                     collapse-clause |
!                     ordered-clause

  !ERROR: When SCHEDULE clause has AUTO specified, it must not have chunk size specified
  !ERROR: At most one SCHEDULE clause can appear on the DO directive
  !ERROR: When SCHEDULE clause has RUNTIME specified, it must not have chunk size specified
  !$omp do schedule(auto, 2) schedule(runtime, 2)
  do i = 1, N
     a = 3.14
  enddo

  !ERROR: A modifier may not be specified in a LINEAR clause on the DO or SIMD directive
  !ERROR: Internal: no symbol found for 'b'
  !$omp do linear(ref(b))
  do i = 1, N
     a = 3.14
  enddo

  !ERROR: The NONMONOTONIC modifier can only be specified with SCHEDULE(DYNAMIC) or SCHEDULE(GUIDED)
  !ERROR: The NONMONOTONIC modifier cannot be specified if an ORDERED clause is specified
  !$omp do schedule(NONMONOTONIC:static) ordered
  do i = 1, N
     a = 3.14
  enddo

  !$omp do schedule(simd, monotonic:dynamic)
  do i = 1, N
     a = 3.14
  enddo

  !ERROR: The parameter of the ORDERED clause must be a constant positive integer expression
  !ERROR: A loop directive may not have both a LINEAR clause and an ORDERED clause with a parameter
  !ERROR: Internal: no symbol found for 'b'
  !ERROR: Internal: no symbol found for 'a'
  !$omp do ordered(1-1) private(b) linear(b) linear(a)
  do i = 1, N
     a = 3.14
  enddo

  !ERROR: The parameter of the ORDERED clause must be greater than or equal to the parameter of the COLLAPSE clause
  !$omp do collapse(num) ordered(1+2+3+4)
  do i = 1, N
     do j = 1, N
        do k = 1, N
           a = 3.14
        enddo
     enddo
  enddo

! 2.8.1 simd-clause -> safelen-clause |
!                      simdlen-clause |
!                      linear-clause |
!                      aligned-clause |
!                      private-clause |
!                      lastprivate-clause |
!                      reduction-clause |
!                      collapse-clause

  a = 0.0
  !$omp simd private(b) reduction(+:a)
  do i = 1, N
     a = a + b + 3.14
  enddo

  !ERROR: At most one SAFELEN clause can appear on the SIMD directive
  !$omp simd safelen(1) safelen(2)
  do i = 1, N
     a = 3.14
  enddo

  !ERROR: The parameter of the SIMDLEN clause must be a constant positive integer expression
  !$omp simd simdlen(-1)
  do i = 1, N
     a = 3.14
  enddo

  !ERROR: The ALIGNMENT parameter of the ALIGNED clause must be a constant positive integer expression
  !ERROR: Internal: no symbol found for 'b'
  !$omp simd aligned(b:-2)
  do i = 1, N
     a = 3.14
  enddo

  !ERROR: The parameter of the SIMDLEN clause must be less than or equal to the parameter of the SAFELEN clause
  !$omp simd safelen(1+1) simdlen(1+2)
  do i = 1, N
     a = 3.14
  enddo
end
