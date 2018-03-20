#ifndef FORTRAN_PARSER_USER_STATE_H_
#define FORTRAN_PARSER_USER_STATE_H_

// Instances of ParseState (parse-state.h) incorporate instances of this
// UserState class, which encapsulates any semantic information necessary for
// parse tree construction so as to avoid any need for representing
// state in static data.

#include "char-block.h"
#include <cinttypes>
#include <set>
#include <unordered_set>

namespace Fortran {
namespace parser {

class UserState {
public:
  using Label = std::uint64_t;
  bool IsDoLabel(Label label) const {
    return doLabels_.find(label) != doLabels_.end();
  }
  bool InNonlabelDoConstruct() const {
    return nonlabelDoConstructNestingDepth_ > 0;
  }
  void NewDoLabel(Label label) { doLabels_.insert(label); }
  void NewSubprogram() {
    doLabels_.clear();
    nonlabelDoConstructNestingDepth_ = 0;
  }
  void EnterNonlabelDoConstruct() { ++nonlabelDoConstructNestingDepth_; }
  void LeaveDoConstruct() {
    if (nonlabelDoConstructNestingDepth_ > 0) {
      --nonlabelDoConstructNestingDepth_;
    }
  }

  void NoteDefinedOperator(const CharBlock &opr) {
    definedOperators_.insert(opr);
  }
  bool IsDefinedOperator(const CharBlock &opr) const {
    return definedOperators_.find(opr) != definedOperators_.end();
  }

private:
  std::unordered_set<Label> doLabels_;
  int nonlabelDoConstructNestingDepth_{0};

  std::set<CharBlock> definedOperators_;
};
}  // namespace parser
}  // namespace Fortran
#endif  // FORTRAN_PARSER_USER_STATE_H_
