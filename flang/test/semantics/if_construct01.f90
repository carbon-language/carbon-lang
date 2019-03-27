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

! Simple check that if constructs are ok.

if (a < b) then
  a = 1
end if

if (a < b) then
  a = 2
else
  a = 3
endif

if (a < b) then
  a = 4
else if(a == b) then
  a = 5
end if

if (a < b) then
  a = 6
else if(a == b) then
  a = 7
elseif(a > b) then
  a = 8
end if

if (a < b) then
  a = 9
else if(a == b) then
  a = 10
else
  a = 11
end if

if (a < b) then
  a = 12
else if(a == b) then
  a = 13
else if(a > b) then
  a = 14
end if

end
