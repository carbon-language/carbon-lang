! Copyright (c) 2019, Arm Ltd.  All rights reserved.
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
! CHECK: exit from DO CONCURRENT construct \\(nc5: do concurrent\\(i5=1:n\\)\\) to construct with name 'mytest1'
! CHECK: exit from DO CONCURRENT construct \\(nc3: do concurrent\\(i3=1:n\\)\\) to construct with name 'mytest1'
! CHECK: exit from DO CONCURRENT construct \\(nc1: do concurrent\\(i1=1:n\\)\\) to construct with name 'mytest1'
! CHECK: exit from DO CONCURRENT construct \\(nc5: do concurrent\\(i5=1:n\\)\\) to construct with name 'nc3'
! CHECK: exit from DO CONCURRENT construct \\(nc3: do concurrent\\(i3=1:n\\)\\) to construct with name 'nc3'
! CHECK: exit from DO CONCURRENT construct \\(nc3: do concurrent\\(i3=1:n\\)\\) to construct with name 'nc2'

subroutine do_concurrent_test1(n)
  implicit none
  integer :: i1,i2,i3,i4,i5,i6,n
  mytest1: if (n>0) then
  nc1:       do concurrent(i1=1:n)
  nc2:         do i2=1,n
  nc3:           do concurrent(i3=1:n)
  nc4:             do i4=1,n
  nc5:               do concurrent(i5=1:n)
  nc6:                 do i6=1,n
                         if (i6==10) exit mytest1
                       end do nc6
                     end do nc5
                   end do nc4
                 end do nc3
               end do nc2
             end do nc1
           end if mytest1
end subroutine do_concurrent_test1

subroutine do_concurrent_test2(n)
  implicit none
  integer :: i1,i2,i3,i4,i5,i6,n
  mytest2: if (n>0) then
  nc1:       do concurrent(i1=1:n)
  nc2:         do i2=1,n
  nc3:           do concurrent(i3=1:n)
  nc4:             do i4=1,n
  nc5:               do concurrent(i5=1:n)
  nc6:                 do i6=1,n
                         if (i6==10) exit nc3
                       end do nc6
                     end do nc5
                   end do nc4
                 end do nc3
               end do nc2
             end do nc1
           end if mytest2
end subroutine do_concurrent_test2

subroutine do_concurrent_test3(n)
  implicit none
  integer :: i1,i2,i3,i4,i5,i6,n
  mytest3: if (n>0) then
  nc1:       do concurrent(i1=1:n)
  nc2:         do i2=1,n
  nc3:           do concurrent(i3=1:n)
                   if (i3==4) exit nc2
  nc4:             do i4=1,n
  nc5:               do concurrent(i5=1:n)
  nc6:                 do i6=1,n
                         if (i6==10) print *, "hello"
                       end do nc6
                     end do nc5
                   end do nc4
                 end do nc3
               end do nc2
             end do nc1
           end if mytest3
end subroutine do_concurrent_test3
