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

#include "parsing.h"
#include "grammar.h"
#include "instrumented-parser.h"
#include "message.h"
#include "openmp-grammar.h"
#include "preprocessor.h"
#include "prescan.h"
#include "provenance.h"
#include "source.h"
#include <sstream>

namespace Fortran::parser {

Parsing::Parsing(AllSources &s) : cooked_{s} {}
Parsing::~Parsing() {}

const SourceFile *Parsing::Prescan(const std::string &path, Options options) {
  options_ = options;
  AllSources &allSources{cooked_.allSources()};
  if (options.isModuleFile) {
    for (const auto &path : options.searchDirectories) {
      allSources.PushSearchPathDirectory(path);
    }
  }

  std::stringstream fileError;
  const SourceFile *sourceFile;
  if (path == "-") {
    sourceFile = allSources.ReadStandardInput(&fileError);
  } else {
    sourceFile = allSources.Open(path, &fileError);
  }
  if (!fileError.str().empty()) {
    ProvenanceRange range{allSources.AddCompilerInsertion(path)};
    messages_.Say(range, "%s"_err_en_US, fileError.str());
    return sourceFile;
  }
  CHECK(sourceFile);

  if (!options.isModuleFile) {
    // For .mod files we always want to look in the search directories.
    // For normal source files we don't push them until after the primary
    // source file has been opened.  If foo.f is missing from the current
    // working directory, we don't want to accidentally read another foo.f
    // from another directory that's on the search path.
    for (const auto &path : options.searchDirectories) {
      allSources.PushSearchPathDirectory(path);
    }
  }

  Preprocessor preprocessor{allSources};
  for (const auto &predef : options.predefinitions) {
    if (predef.second) {
      preprocessor.Define(predef.first, *predef.second);
    } else {
      preprocessor.Undefine(predef.first);
    }
  }
  Prescanner prescanner{messages_, cooked_, preprocessor, options.features};
  prescanner.set_fixedForm(options.isFixedForm)
      .set_fixedFormColumnLimit(options.fixedFormColumns)
      .AddCompilerDirectiveSentinel("dir$");
  if (options.features.IsEnabled(LanguageFeature::OpenMP)) {
    prescanner.AddCompilerDirectiveSentinel("$omp");
    prescanner.AddCompilerDirectiveSentinel("$");  // OMP conditional line
  }
  ProvenanceRange range{allSources.AddIncludedFile(
      *sourceFile, ProvenanceRange{}, options.isModuleFile)};
  prescanner.Prescan(range);
  if (cooked_.BufferedBytes() == 0 && !options.isModuleFile) {
    // Input is empty.  Append a newline so that any warning
    // message about nonstandard usage will have provenance.
    cooked_.Put('\n', range.start());
  }
  cooked_.Marshal();
  if (options.needProvenanceRangeToCharBlockMappings) {
    cooked_.CompileProvenanceRangeToOffsetMappings();
  }
  return sourceFile;
}

void Parsing::DumpCookedChars(std::ostream &out) const {
  UserState userState{cooked_, common::LanguageFeatureControl{}};
  ParseState parseState{cooked_};
  parseState.set_inFixedForm(options_.isFixedForm).set_userState(&userState);
  while (std::optional<const char *> p{parseState.GetNextChar()}) {
    out << **p;
  }
}

void Parsing::DumpProvenance(std::ostream &out) const { cooked_.Dump(out); }

void Parsing::DumpParsingLog(std::ostream &out) const {
  log_.Dump(out, cooked_);
}

void Parsing::Parse(std::ostream *out) {
  UserState userState{cooked_, options_.features};
  userState.set_debugOutput(out)
      .set_instrumentedParse(options_.instrumentedParse)
      .set_log(&log_);
  ParseState parseState{cooked_};
  parseState.set_inFixedForm(options_.isFixedForm).set_userState(&userState);
  parseTree_ = program.Parse(parseState);
  CHECK(
      !parseState.anyErrorRecovery() || parseState.messages().AnyFatalError());
  consumedWholeFile_ = parseState.IsAtEnd();
  messages_.Annex(std::move(parseState.messages()));
  finalRestingPlace_ = parseState.GetLocation();
}

void Parsing::ClearLog() { log_.clear(); }

bool Parsing::ForTesting(std::string path, std::ostream &err) {
  Prescan(path, Options{});
  if (messages_.AnyFatalError()) {
    messages_.Emit(err, cooked_);
    err << "could not scan " << path << '\n';
    return false;
  }
  Parse();
  messages_.Emit(err, cooked_);
  if (!consumedWholeFile_) {
    EmitMessage(err, finalRestingPlace_, "parser FAIL; final position");
    return false;
  }
  if (messages_.AnyFatalError() || !parseTree_.has_value()) {
    err << "could not parse " << path << '\n';
    return false;
  }
  return true;
}
}
