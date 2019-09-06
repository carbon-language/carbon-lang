// Copyright (c) 2018-2019, NVIDIA CORPORATION.  All rights reserved.
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

#ifndef FORTRAN_PARSER_PARSING_H_
#define FORTRAN_PARSER_PARSING_H_

#include "characters.h"
#include "features.h"
#include "instrumented-parser.h"
#include "message.h"
#include "parse-tree.h"
#include "provenance.h"
#include <optional>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

namespace Fortran::parser {

struct Options {
  Options() {}

  using Predefinition = std::pair<std::string, std::optional<std::string>>;

  bool isFixedForm{false};
  int fixedFormColumns{72};
  LanguageFeatureControl features;
  std::vector<std::string> searchDirectories;
  std::vector<Predefinition> predefinitions;
  bool instrumentedParse{false};
  bool isModuleFile{false};
  bool needProvenanceRangeToCharBlockMappings{false};
};

class Parsing {
public:
  explicit Parsing(AllSources &);
  ~Parsing();

  bool consumedWholeFile() const { return consumedWholeFile_; }
  const char *finalRestingPlace() const { return finalRestingPlace_; }
  CookedSource &cooked() { return cooked_; }
  Messages &messages() { return messages_; }
  std::optional<Program> &parseTree() { return parseTree_; }

  void Prescan(const std::string &path, Options);
  void DumpCookedChars(std::ostream &) const;
  void DumpProvenance(std::ostream &) const;
  void DumpParsingLog(std::ostream &) const;
  void Parse(std::ostream *debugOutput = nullptr);
  void ClearLog();

  void EmitMessage(std::ostream &o, const char *at, const std::string &message,
      bool echoSourceLine = false) const {
    cooked_.allSources().EmitMessage(
        o, cooked_.GetProvenanceRange(CharBlock(at)), message, echoSourceLine);
  }

  bool ForTesting(std::string path, std::ostream &);

private:
  Options options_;
  CookedSource cooked_;
  Messages messages_;
  bool consumedWholeFile_{false};
  const char *finalRestingPlace_{nullptr};
  std::optional<Program> parseTree_;
  ParsingLog log_;
};
}
#endif  // FORTRAN_PARSER_PARSING_H_
