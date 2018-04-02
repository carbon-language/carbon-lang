#include "parsing.h"
#include "grammar.h"
#include "message.h"
#include "preprocessor.h"
#include "prescan.h"
#include "provenance.h"
#include "source.h"
#include <sstream>

namespace Fortran {
namespace parser {

void Parsing::Prescan(const std::string &path, Options options) {
  options_ = options;

  std::stringstream fileError;
  const auto *sourceFile = allSources_.Open(path, &fileError);
  if (sourceFile == nullptr) {
    ProvenanceRange range{allSources_.AddCompilerInsertion(path)};
    MessageFormattedText msg("%s"_err_en_US, fileError.str().data());
    messages_.Put(Message(range.start(), std::move(msg)));
    return;
  }
  if (sourceFile->bytes() == 0) {
    ProvenanceRange range{allSources_.AddCompilerInsertion(path)};
    messages_.Put(Message{range.start(), "file is empty"_err_en_US});
    return;
  }

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
  ProvenanceRange range{
      allSources_.AddIncludedFile(*sourceFile, ProvenanceRange{})};
  prescanner.Prescan(range);
  cooked_.Marshal();
}

void Parsing::DumpCookedChars(std::ostream &out) const {
  UserState userState;
  ParseState parseState{cooked_};
  parseState.set_inFixedForm(options_.isFixedForm).set_userState(&userState);
  while (std::optional<char> ch{parseState.GetNextChar()}) {
    out << *ch;
  }
}

void Parsing::DumpProvenance(std::ostream &out) const { cooked_.Dump(out); }

void Parsing::Parse() {
  UserState userState;
  ParseState parseState{cooked_};
  parseState.set_inFixedForm(options_.isFixedForm)
      .set_encoding(options_.encoding)
      .set_warnOnNonstandardUsage(options_.isStrictlyStandard)
      .set_warnOnDeprecatedUsage(options_.isStrictlyStandard)
      .set_userState(&userState);
  parseTree_ = program.Parse(&parseState);
  CHECK(
      !parseState.anyErrorRecovery() || parseState.messages()->AnyFatalError());
  consumedWholeFile_ = parseState.IsAtEnd();
  finalRestingPlace_ = parseState.GetLocation();
  messages_.Annex(parseState.messages());
}

bool Parsing::ForTesting(std::string path, std::ostream &err) {
  Prescan(path, Options{});
  if (messages_.AnyFatalError()) {
    messages_.Emit(err);
    err << "could not scan " << path << '\n';
    return false;
  }
  Parse();
  messages_.Emit(err);
  if (!consumedWholeFile_) {
    err << "f18 parser FAIL; final position: ";
    Identify(err, finalRestingPlace_, "   ");
    return false;
  }
  if (messages_.AnyFatalError() || !parseTree_.has_value()) {
    err << "could not parse " << path << '\n';
    return false;
  }
  return true;
}
}  // namespace parser
}  // namespace Fortran
