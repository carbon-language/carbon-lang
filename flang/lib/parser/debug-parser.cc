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

#include "debug-parser.h"
#include "user-state.h"
#include <ostream>
#include <string>

namespace Fortran {
namespace parser {

std::optional<Success> DebugParser::Parse(ParseState &state) const {
  if (auto ustate = state.userState()) {
    if (auto out = ustate->debugOutput()) {
      const CookedSource &cooked{ustate->cooked()};
      if (auto context = state.context()) {
        context->Emit(*out, cooked);
      }
      Provenance p{cooked.GetProvenance(state.GetLocation()).start()};
      cooked.allSources().Identify(*out, p, "", true);
      *out << "   parser debug: " << std::string{str_, length_} << "\n\n";
    }
  }
  return {Success{}};
}
}  // namespace parser
}  // namespace Fortran
