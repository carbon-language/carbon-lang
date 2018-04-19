#ifndef FORTRAN_PARSER_INSTRUMENTED_PARSER_H_
#define FORTRAN_PARSER_INSTRUMENTED_PARSER_H_

#include "message.h"
#include "parse-state.h"
#include "user-state.h"
#include <cstddef>
#include <map>
#include <ostream>

namespace Fortran {
namespace parser {

class ParsingLog {
public:
  void Note(const char *at, const MessageFixedText &tag, bool pass);
  void Dump(std::ostream &) const;

private:
  struct LogForPosition {
    struct Entries {
      int passes{0};
      int failures{0};
    };
    std::map<MessageFixedText, Entries> perTag;
  };
  std::map<std::size_t, LogForPosition> perPos_;
};

template<typename PA> class InstrumentedParser {
public:
  using resultType = typename PA::resultType;
  constexpr InstrumentedParser(const InstrumentedParser &) = default;
  constexpr InstrumentedParser(const MessageFixedText &tag, const PA &parser)
    : tag_{tag}, parser_{parser} {}
  std::optional<resultType> Parse(ParseState *state) const {
    const char *at{state->GetLocation()};
    std::optional<resultType> result{parser_.Parse(state)};
    if (UserState * ustate{state->userState()}) {
      if (ParsingLog * log{ustate->log()}) {
        log->Note(at, tag_, result.has_value());
      }
    }
    return result;
  }

private:
  const MessageFixedText tag_;
  const PA parser_;
};

template<typename PA>
inline constexpr auto instrumented(
    const MessageFixedText &tag, const PA &parser) {
  return InstrumentedParser{tag, parser};
}
}  // namespace parser
}  // namespace Fortran
#endif  // FORTRAN_PARSER_INSTRUMENTED_PARSER_H_
