#include "instrumented-parser.h"
#include "message.h"
#include "provenance.h"
#include <map>
#include <ostream>

namespace Fortran {
namespace parser {

void ParsingLog::clear() { perPos_.clear(); }

// In the logs, just use the addresses of the message texts to sort the
// map keys.
bool operator<(const MessageFixedText &x, const MessageFixedText &y) {
  return x.str() < y.str();
}

bool ParsingLog::Fails(
    const char *at, const MessageFixedText &tag, ParseState &state) {
  std::size_t offset = reinterpret_cast<std::size_t>(at);
  auto posIter = perPos_.find(offset);
  if (posIter == perPos_.end()) {
    return false;
  }
  auto tagIter = posIter->second.perTag.find(tag);
  if (tagIter == posIter->second.perTag.end()) {
    return false;
  }
  auto &entry = tagIter->second;
  if (entry.deferred && !state.deferMessages()) {
    return false;  // don't fail fast, we want to generate messages
  }
  ++entry.count;
  if (!state.deferMessages()) {
    state.messages().Copy(entry.messages);
  }
  return !entry.pass;
}

void ParsingLog::Note(const char *at, const MessageFixedText &tag, bool pass,
    const ParseState &state) {
  std::size_t offset = reinterpret_cast<std::size_t>(at);
  auto &entry = perPos_[offset].perTag[tag];
  if (++entry.count == 1) {
    entry.pass = pass;
    entry.deferred = state.deferMessages();
    if (!entry.deferred) {
      entry.messages.Copy(state.messages());
    }
  } else {
    CHECK(entry.pass == pass);
    if (entry.deferred && !state.deferMessages()) {
      entry.deferred = false;
      entry.messages.Copy(state.messages());
    }
  }
}

void ParsingLog::Dump(std::ostream &o, const CookedSource &cooked) const {
  for (const auto &posLog : perPos_) {
    const char *at{reinterpret_cast<const char *>(posLog.first)};
    for (const auto &tagLog : posLog.second.perTag) {
      Message{at, tagLog.first}.Emit(o, cooked, true);
      auto &entry = tagLog.second;
      o << "  " << (entry.pass ? "pass" : "fail") << " " << entry.count << '\n';
      entry.messages.Emit(o, cooked, "      ");
    }
  }
}
}  // namespace parser
}  // namespace Fortran
