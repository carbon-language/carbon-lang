#include "instrumented-parser.h"
#include "message.h"
#include <map>
#include <ostream>

namespace Fortran {
namespace parser {

// In the logs, just use the addresses of the message texts to sort the
// map keys.
bool operator<(const MessageFixedText &x, const MessageFixedText &y) {
  return x.str() < y.str();
}

void ParsingLog::Note(const char *at, const MessageFixedText &tag, bool pass) {
  std::size_t offset = reinterpret_cast<std::size_t>(at);
  if (pass) {
    ++perPos_[offset].perTag[tag].passes;
  } else {
    ++perPos_[offset].perTag[tag].failures;
  }
}

void ParsingLog::Dump(std::ostream &o) const {
  for (const auto &posLog : perPos_) {
    o << "at offset " << posLog.first << ":\n";
    for (const auto &tagLog : posLog.second.perTag) {
      o << "  " << tagLog.first.ToString() << ' ' << tagLog.second.passes
        << ", " << tagLog.second.failures << '\n';
    }
  }
}
}  // namespace parser
}  // namespace Fortran
