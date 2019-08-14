# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

add_executable(quick-sanity-test
  quick-sanity-test.cc
)

target_link_libraries(quick-sanity-test
  FortranDecimal
)

# add_executable(thorough-test
#   thorough-test.cc
# )

# target_link_libraries(thorough-test
#   FortranDecimal
# )

# add_executable(benchmark01
#   benchmark01.cc
# )

# target_link_libraries(benchmark01
#   FortranDecimal
# )

add_test(Sanity quick-sanity-test)
