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

#ifndef FORTRAN_SEMANTICS_MOD_FILE_H_
#define FORTRAN_SEMANTICS_MOD_FILE_H_

#include "resolve-names.h"
#include "../parser/char-block.h"
#include "../parser/message.h"
#include "../parser/parse-tree.h"
#include "../parser/parsing.h"
#include "../parser/provenance.h"
#include <iostream>
#include <string>

namespace Fortran::semantics {

using SourceName = parser::CharBlock;

void WriteModFiles();

class ModFileReader {
public:
  // directories specifies where to search for module files
  ModFileReader(const std::vector<std::string> &directories)
    : directories_{directories} {}

  // Find and read the module file for modName.
  // Return true on success; otherwise errors() reports the problems.
  bool Read(const SourceName &modName);
  std::list<parser::Message> &errors() { return errors_; }

private:
  std::vector<std::string> directories_;
  parser::AllSources allSources_;
  std::unique_ptr<parser::CookedSource> cooked_{
      std::make_unique<parser::CookedSource>(allSources_)};
  std::list<parser::Message> errors_;

  std::optional<std::string> FindModFile(const SourceName &);
  bool Prescan(const SourceName &, const std::string &);
};

}  // namespace Fortran::semantics

#endif
