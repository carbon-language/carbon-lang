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

! Check OpenMP declarative directives

!TODO: all internal errors
!      enable declare-reduction example after name resolution

! 2.8.2 declare-simd

subroutine declare_simd_1(a, b)
  real(8), intent(inout) :: a, b
  !ERROR: Internal: no symbol found for 'declare_simd_1'
  !$omp declare simd(declare_simd_1)
  a = 3.14 + b
end subroutine declare_simd_1

module m1
  abstract interface
     subroutine sub(x,y)
       integer, intent(in)::x
       integer, intent(in)::y
     end subroutine sub
  end interface
end module m1

subroutine declare_simd_2
  use m1
  procedure (sub) sub1
  !ERROR: Internal: no symbol found for 'sub1'
  !$omp declare simd(sub1)
  procedure (sub), pointer::p
  p=>sub1
  call p(5,10)
end subroutine declare_simd_2

subroutine sub1 (x,y)
  integer, intent(in)::x, y
  print *, x+y
end subroutine sub1

! 2.10.6 declare-target
! 2.15.2 threadprivate

module m2
  !$omp declare target
  !$omp declare target (N)
  !$omp declare target to(Q)
  integer, parameter :: N=10000, M=1024
  integer :: i
  real :: Q(N, N)
  !$omp threadprivate(i)

contains
  subroutine foo
  end subroutine foo
end module m2

! 2.16 declare-reduction

! subroutine declare_red_1()
!   use omp_lib
!   integer :: my_var
!   !$omp declare reduction (my_add_red : integer : omp_out = omp_out + omp_in) initializer (omp_priv=0)
!   my_var = 0
!   !$omp parallel reduction (my_add_red : my_var) num_threads(4)
!   my_var = omp_get_thread_num() + 1
!   !$omp end parallel
!   print *, "sum of thread numbers is ", my_var
! end subroutine declare_red_1

end
