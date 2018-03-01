#ifndef FORTRAN_PARSER_PARSING_H_
#define FORTRAN_PARSER_PARSING_H_

#include "characters.h"
#include "message.h"
#include "parse-tree.h"
#include "provenance.h"
#include <ostream>

namespace Fortran {
namespace parser {

struct Options {
  Options() {}

  bool isFixedForm{false};
  int fixedFormColumns{72};
  bool enableBackslashEscapes{true};
  bool enableOldDebugLines{false};
  bool isStrictlyStandard{false};
  Encoding encoding{Encoding::UTF8};
};

class Parsing {
public:
  Parsing() {}

  bool consumedWholeFile() const { return consumedWholeFile_; }
  Provenance finalRestingPlace() const { return finalRestingPlace_; }
  Messages &messages() { return messages_; }
  Program &parseTree() { return *parseTree_; }

  void PushSearchPathDirectory(std::string);
  bool Prescan(const std::string &path, Options);
  void DumpCookedChars(std::ostream &) const;
  void DumpProvenance(std::ostream &) const;
  bool Parse();

  void Identify(std::ostream &o, Provenance p, const std::string &prefix,
      bool echoSourceLine = false) const {
    allSources_.Identify(o, p, prefix, echoSourceLine);
  }

private:
  Options options_;
  AllSources allSources_;
  Messages messages_{allSources_};
  CookedSource cooked_{&allSources_};
  bool anyFatalError_{false};
  bool consumedWholeFile_{false};
  Provenance finalRestingPlace_;
  std::optional<Program> parseTree_;
};
}  // namespace parser
}  // namespace Fortran
#endif  // FORTRAN_PARSER_PARSING_H_
