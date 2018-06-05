// Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "real.h"

namespace Fortran::evaluate {

template class Real<Integer<16>, 11>;
template class Real<Integer<32>, 24>;
template class Real<Integer<64>, 53>;
template class Real<Integer<80>, 64, false>;
template class Real<Integer<128>, 112>;

}  // namespace Fortran::evaluate
