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

#include "parsing.h"
#include "grammar.h"
#include "instrumented-parser.h"
#include "message.h"
#include "preprocessor.h"
#include "prescan.h"
#include "provenance.h"
#include "source.h"
#include <sstream>

namespace Fortran::parser {

void Parsing::Prescan(const std::string &path, Options options) {
  options_ = options;

  std::stringstream fileError;
  const SourceFile *sourceFile;
  if (path == "-") {
    sourceFile = allSources_.ReadStandardInput(&fileError);
  } else {
    sourceFile = allSources_.Open(path, &fileError);
  }
  if (sourceFile == nullptr) {
    ProvenanceRange range{allSources_.AddCompilerInsertion(path)};
    MessageFormattedText msg("%s"_err_en_US, fileError.str().data());
    messages_.Put(Message{range, std::move(msg)});
    return;
  }
  if (sourceFile->bytes() == 0) {
    ProvenanceRange range{allSources_.AddCompilerInsertion(path)};
    messages_.Put(Message{range, "file is empty"_err_en_US});
    return;
  }

  // N.B. Be sure to not push the search directory paths until the primary
  // source file has been opened.  If foo.f is missing from the current
  // working directory, we don't want to accidentally read another foo.f
  // from another directory that's on the search path.
  for (const auto &path : options.searchDirectories) {
    allSources_.PushSearchPathDirectory(path);
  }

  Preprocessor preprocessor{allSources_};
  for (const auto &predef : options.predefinitions) {
    if (predef.second.has_value()) {
      preprocessor.Define(predef.first, *predef.second);
    } else {
      preprocessor.Undefine(predef.first);
    }
  }
  Prescanner prescanner{messages_, cooked_, preprocessor};
  prescanner.set_fixedForm(options.isFixedForm)
      .set_fixedFormColumnLimit(options.fixedFormColumns)
      .set_encoding(options.encoding)
      .set_enableBackslashEscapesInCharLiterals(options.enableBackslashEscapes)
      .set_enableOldDebugLines(options.enableOldDebugLines)
      .set_warnOnNonstandardUsage(options_.isStrictlyStandard)
      .AddCompilerDirectiveSentinel("dir$");
  if (options.enableOpenMP) {
    prescanner.AddCompilerDirectiveSentinel("$omp");
  }
  ProvenanceRange range{
      allSources_.AddIncludedFile(*sourceFile, ProvenanceRange{})};
  prescanner.Prescan(range);
  cooked_.Marshal();
}

void Parsing::DumpCookedChars(std::ostream &out) const {
  UserState userState{cooked_};
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
  UserState userState{cooked_};
  userState.set_debugOutput(out)
      .set_instrumentedParse(options_.instrumentedParse)
      .set_log(&log_);
  ParseState parseState{cooked_};
  parseState.set_inFixedForm(options_.isFixedForm)
      .set_encoding(options_.encoding)
      .set_warnOnNonstandardUsage(options_.isStrictlyStandard)
      .set_warnOnDeprecatedUsage(options_.isStrictlyStandard)
      .set_userState(&userState);
  parseTree_ = program.Parse(parseState);
  CHECK(
      !parseState.anyErrorRecovery() || parseState.messages().AnyFatalError());
  consumedWholeFile_ = parseState.IsAtEnd();
  messages_.Annex(parseState.messages());
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

}  // namespace Fortran::parser
