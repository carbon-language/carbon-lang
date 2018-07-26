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

#include "attr.h"
#include "resolve-names.h"
#include "../parser/message.h"
#include "../parser/provenance.h"
#include <iosfwd>
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace Fortran::parser {
class CharBlock;
}

namespace Fortran::semantics {

using SourceName = parser::CharBlock;
class Symbol;
class Scope;

class ModFileWriter {
public:
  // The .mod file format version number.
  void set_version(int version) { version_ = version; }
  // The directory to write .mod files in.
  void set_directory(const std::string &dir) { dir_ = dir; }

  // Errors encountered during writing. Non-empty if WriteAll returns false.
  const std::vector<parser::MessageFormattedText> &errors() const {
    return errors_;
  }

  // Write out all .mod files; if error return false.
  bool WriteAll();
  // Write out .mod file for one module; if error return false.
  bool WriteOne(const Symbol &);

private:
  using symbolSet = std::set<const Symbol *>;
  using symbolVector = std::vector<const Symbol *>;

  int version_{1};
  std::string dir_{"."};
  // The mod file consists of uses, declarations, and contained subprograms:
  std::stringstream uses_;
  std::stringstream useExtraAttrs_;  // attrs added to used entity
  std::stringstream decls_;
  std::stringstream contains_;
  // Any errors encountered during writing:
  std::vector<parser::MessageFormattedText> errors_;

  std::string GetAsString(const std::string &);
  std::string GetHeader(const std::string &);
  void PutSymbols(const Scope &);
  symbolVector SortSymbols(const symbolSet);
  symbolSet CollectSymbols(const Scope &);
  void PutSymbol(const Symbol &);
  void PutDerivedType(const Symbol &);
  void PutSubprogram(const Symbol &);
  void PutGeneric(const Symbol &);
  void PutUse(const Symbol &);
  void PutUseExtraAttr(Attr, const Symbol &, const Symbol &);
};

class ModFileReader {
public:
  // directories specifies where to search for module files
  ModFileReader(const std::vector<std::string> &directories)
    : directories_{directories} {}

  // Find and read the module file for modName.
  // Return true on success; otherwise errors() reports the problems.
  bool Read(const SourceName &modName);
  std::vector<parser::Message> &errors() { return errors_; }

private:
  std::vector<std::string> directories_;
  std::vector<parser::Message> errors_;

  std::optional<std::string> FindModFile(const SourceName &);
  bool Prescan(const SourceName &, const std::string &);
};

}  // namespace Fortran::semantics

#endif
