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

#ifndef FORTRAN_TEST_EVALUATE_FP_TESTING_H_
#define FORTRAN_TEST_EVALUATE_FP_TESTING_H_

#include "../../lib/evaluate/common.h"
#include <fenv.h>

using Fortran::evaluate::RealFlags;

class ScopedHostFloatingPointEnvironment {
public:
  ScopedHostFloatingPointEnvironment(bool treatDenormalOperandsAsZero = false,
                                     bool flushDenormalResultsToZero = false);
  ~ScopedHostFloatingPointEnvironment();
  RealFlags CurrentFlags() const;
private:
  fenv_t originalFenv_;
  fenv_t currentFenv_;
};

#endif  // FORTRAN_TEST_EVALUATE_FP_TESTING_H_
