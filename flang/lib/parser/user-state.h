#ifndef FORTRAN_PARSER_USER_STATE_H_
#define FORTRAN_PARSER_USER_STATE_H_

// Instances of ParseState (parse-state.h) incorporate instances of this
// UserState class, which encapsulates any semantic information necessary for
// parse tree construction so as to avoid any need for representing
// state in static data.

#include "basic-parsers.h"
#include "char-block.h"
#include "parse-tree.h"
#include <cinttypes>
#include <optional>
#include <set>
#include <unordered_set>

namespace Fortran {
namespace parser {

class CookedSource;
class ParsingLog;
class ParseState;

class UserState {
public:
  explicit UserState(const CookedSource &cooked) : cooked_{cooked} {}

  const CookedSource &cooked() const { return cooked_; }

  ParsingLog *log() const { return log_; }
  UserState &set_log(ParsingLog *log) {
    log_ = log;
    return *this;
  }

  bool instrumentedParse() const { return instrumentedParse_; }
  UserState &set_instrumentedParse(bool yes) {
    instrumentedParse_ = yes;
    return *this;
  }

  void NewSubprogram() {
    doLabels_.clear();
    nonlabelDoConstructNestingDepth_ = 0;
    oldStructureComponents_.clear();
  }

  using Label = std::uint64_t;
  bool IsDoLabel(Label label) const {
    return doLabels_.find(label) != doLabels_.end();
  }
  bool InNonlabelDoConstruct() const {
    return nonlabelDoConstructNestingDepth_ > 0;
  }
  void NewDoLabel(Label label) { doLabels_.insert(label); }

  void EnterNonlabelDoConstruct() { ++nonlabelDoConstructNestingDepth_; }
  void LeaveDoConstruct() {
    if (nonlabelDoConstructNestingDepth_ > 0) {
      --nonlabelDoConstructNestingDepth_;
    }
  }

  void NoteOldStructureComponent(const CharBlock &name) {
    oldStructureComponents_.insert(name);
  }
  bool IsOldStructureComponent(const CharBlock &name) const {
    return oldStructureComponents_.find(name) != oldStructureComponents_.end();
  }

private:
  const CookedSource &cooked_;

  ParsingLog *log_{nullptr};
  bool instrumentedParse_{false};

  std::unordered_set<Label> doLabels_;
  int nonlabelDoConstructNestingDepth_{0};

  std::set<CharBlock> oldStructureComponents_;
};

// Definitions of parser classes that manipulate the UserState.
struct StartNewSubprogram {
  using resultType = Success;
  static std::optional<Success> Parse(ParseState &);
};

struct CapturedLabelDoStmt {
  using resultType = Statement<Indirection<LabelDoStmt>>;
  static std::optional<resultType> Parse(ParseState &);
};

struct EndDoStmtForCapturedLabelDoStmt {
  using resultType = Statement<Indirection<EndDoStmt>>;
  static std::optional<resultType> Parse(ParseState &);
};

struct EnterNonlabelDoConstruct {
  using resultType = Success;
  static std::optional<Success> Parse(ParseState &);
};

struct LeaveDoConstruct {
  using resultType = Success;
  static std::optional<Success> Parse(ParseState &);
};

struct OldStructureComponentName {
  using resultType = Name;
  static std::optional<Name> Parse(ParseState &);
};

struct StructureComponents {
  using resultType = DataComponentDefStmt;
  static std::optional<DataComponentDefStmt> Parse(ParseState &);
};
}  // namespace parser
}  // namespace Fortran
#endif  // FORTRAN_PARSER_USER_STATE_H_
