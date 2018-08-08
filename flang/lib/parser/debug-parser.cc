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

namespace Fortran::parser {

std::optional<Success> DebugParser::Parse(ParseState &state) const {
  if (auto ustate{state.userState()}) {
    if (auto out{ustate->debugOutput()}) {
      std::string note{str_, length_};
      Message message{
          state.GetLocation(), "parser debug: %s"_en_US, note.data()};
      message.SetContext(state.context().get());
      message.Emit(*out, ustate->cooked(), true);
    }
  }
  return {Success{}};
}

}  // namespace Fortran::parser
