#ifndef FORTRAN_PARSER_PARSING_H_
#define FORTRAN_PARSER_PARSING_H_

#include "characters.h"
#include "message.h"
#include "parse-tree.h"
#include "provenance.h"
#include <optional>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

namespace Fortran {
namespace parser {

struct Options {
  Options() {}

  using Predefinition = std::pair<std::string, std::optional<std::string>>;

  bool isFixedForm{false};
  int fixedFormColumns{72};
  bool enableBackslashEscapes{true};
  bool enableOldDebugLines{false};
  bool isStrictlyStandard{false};
  Encoding encoding{Encoding::UTF8};
  std::vector<std::string> searchDirectories;
  std::vector<Predefinition> predefinitions;
};

class Parsing {
public:
  Parsing() {}

  bool consumedWholeFile() const { return consumedWholeFile_; }
  const char *finalRestingPlace() const { return finalRestingPlace_; }
  Messages &messages() { return messages_; }
  std::optional<Program> &parseTree() { return parseTree_; }

  void Prescan(const std::string &path, Options);
  void DumpCookedChars(std::ostream &) const;
  void DumpProvenance(std::ostream &) const;
  void Parse();

  void Identify(std::ostream &o, const char *at, const std::string &prefix,
      bool echoSourceLine = false) const {
    allSources_.Identify(
        o, cooked_.GetProvenance(at).start(), prefix, echoSourceLine);
  }

  bool ForTesting(std::string path, std::ostream &);

private:
  Options options_;
  AllSources allSources_;
  CookedSource cooked_{allSources_};
  Messages messages_{cooked_};
  bool consumedWholeFile_{false};
  const char *finalRestingPlace_{nullptr};
  std::optional<Program> parseTree_;
};
}  // namespace parser
}  // namespace Fortran
#endif  // FORTRAN_PARSER_PARSING_H_
