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

void Parsing::PushSearchPathDirectory(std::string path) {
  allSources_.PushSearchPathDirectory(path);
}

bool Parsing::Prescan(const std::string &path, Options options) {
  options_ = options;

  std::stringstream fileError;
  const auto *sourceFile = allSources_.Open(path, &fileError);
  if (sourceFile == nullptr) {
    ProvenanceRange range{allSources_.AddCompilerInsertion(path)};
    MessageFormattedText msg("%s"_en_US, fileError.str().data());
    messages_.Put(Message(range.start(), std::move(msg)));
    anyFatalError_ = true;
    return false;
  }

  Preprocessor preprocessor{&allSources_};
  Prescanner prescanner{&messages_, &cooked_, &preprocessor};
  prescanner.set_fixedForm(options.isFixedForm)
      .set_fixedFormColumnLimit(options.fixedFormColumns)
      .set_encoding(options.encoding)
      .set_enableBackslashEscapesInCharLiterals(options.enableBackslashEscapes)
      .set_enableOldDebugLines(options.enableOldDebugLines);
  ProvenanceRange range{
      allSources_.AddIncludedFile(*sourceFile, ProvenanceRange{})};
  anyFatalError_ = !prescanner.Prescan(range);
  if (anyFatalError_) {
    return false;
  }

  cooked_.Marshal();
  return true;
}

void Parsing::DumpCookedChars(std::ostream &out) const {
  if (anyFatalError_) {
    return;
  }
  UserState userState;
  ParseState parseState{cooked_};
  parseState.set_inFixedForm(options_.isFixedForm).set_userState(&userState);
  while (std::optional<char> ch{parseState.GetNextChar()}) {
    out << *ch;
  }
}

void Parsing::DumpProvenance(std::ostream &out) const { cooked_.Dump(out); }

bool Parsing::Parse() {
  if (anyFatalError_) {
    return false;
  }
  UserState userState;
  ParseState parseState{cooked_};
  parseState.set_inFixedForm(options_.isFixedForm)
      .set_encoding(options_.encoding)
      .set_warnOnNonstandardUsage(options_.isStrictlyStandard)
      .set_warnOnDeprecatedUsage(options_.isStrictlyStandard)
      .set_userState(&userState);
  parseTree_ = program.Parse(&parseState);
  anyFatalError_ = parseState.anyErrorRecovery();
#if 0  // pgf90 -Mstandard enables warnings only, they aren't fatal.
    // TODO: -Werror
    || (options_.isStrictlyStandard && parseState.anyConformanceViolation());
#endif
  consumedWholeFile_ = parseState.IsAtEnd();
  finalRestingPlace_ = parseState.GetProvenance();
  messages_.Annex(parseState.messages());
  return parseTree_.has_value() && !anyFatalError_;
}
}  // namespace parser
}  // namespace Fortran
