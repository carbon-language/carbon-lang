//===--- LRTable.h - Define LR Parsing Table ---------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  The LRTable (referred as LR parsing table in the LR literature) is the core
//  component in LR parsers, it drives the LR parsers by specifying an action to
//  take given the current state on the top of the stack and the current
//  lookahead token.
//
//  The LRTable can be described as a matrix where the rows represent
//  the states of the LR graph, the columns represent the symbols of the
//  grammar, and each entry of the matrix (called action) represents a
//  state transition in the graph.
//
//  Typically, based on the category of the grammar symbol, the LRTable is
//  broken into two logically separate tables:
//    - ACTION table with terminals as columns -- e.g. ACTION[S, a] specifies
//      next action (shift/reduce) on state S under a lookahead terminal a
//    - GOTO table with nonterminals as columns -- e.g. GOTO[S, X] specifies
//      the state which we transist to from the state S with the nonterminal X
//
//  LRTable is *performance-critial* as it is consulted frequently during a
//  parse. In general, LRTable is very sparse (most of the entries are empty).
//  For example, for the C++ language, the SLR table has ~1500 states and 650
//  symbols which results in a matrix having 975K entries, ~90% of entries are
//  empty.
//
//  This file implements a speed-and-space-efficient LRTable.
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_PSEUDO_GRAMMAR_LRTABLE_H
#define CLANG_PSEUDO_GRAMMAR_LRTABLE_H

#include "clang-pseudo/grammar/Grammar.h"
#include "llvm/ADT/ArrayRef.h"
#include <cstdint>
#include <vector>

namespace clang {
namespace pseudo {

// Represents the LR parsing table, which can efficiently the question "what is
// the next step given the lookahead token and current state on top of the
// stack?".
//
// This is a dense implementation, which only takes an amount of space that is
// proportional to the number of non-empty entries in the table.
//
// Unlike the typical LR parsing table which allows at most one available action
// per entry, conflicted actions are allowed in LRTable. The LRTable is designed
// to be used in nondeterministic LR parsers (e.g. GLR).
class LRTable {
public:
  // StateID is only 13 bits wide.
  using StateID = uint16_t;
  static constexpr unsigned StateBits = 13;

  // Action represents the terminal and nonterminal actions, it combines the
  // entry of the ACTION and GOTO tables from the LR literature.
  class Action {
  public:
    enum Kind : uint8_t {
      Sentinel = 0,
      // Terminal actions, corresponding to entries of ACTION table.

      // Shift to state n: move forward with the lookahead, and push state n
      // onto the state stack.
      // A shift is a forward transition, and the value n is the next state that
      // the parser is to enter.
      Shift,
      // Reduce by a rule: pop the state stack.
      Reduce,

      // NOTE: there are no typical accept actions in the LRtable, accept
      // actions are handled specifically in the parser -- if the parser
      // reaches to a target state (which is goto(StartState, StartSymbol)) at
      // the EOF token after a reduce, this indicates the input has been parsed
      // as the StartSymbol successfully.

      // Nonterminal actions, corresponding to entry of GOTO table.

      // Go to state n: push state n onto the state stack.
      // Similar to Shift, but it is a nonterminal forward transition.
      GoTo,
    };

    static Action goTo(StateID S) { return Action(GoTo, S); }
    static Action shift(StateID S) { return Action(Shift, S); }
    static Action reduce(RuleID RID) { return Action(Reduce, RID); }
    static Action sentinel() { return Action(Sentinel, 0); }

    StateID getShiftState() const {
      assert(kind() == Shift);
      return Value;
    }
    StateID getGoToState() const {
      assert(kind() == GoTo);
      return Value;
    }
    RuleID getReduceRule() const {
      assert(kind() == Reduce);
      return Value;
    }
    Kind kind() const { return static_cast<Kind>(K); }

    bool operator==(const Action &L) const { return opaque() == L.opaque(); }
    uint16_t opaque() const { return K << ValueBits | Value; };

  private:
    Action(Kind K1, unsigned Value) : K(K1), Value(Value) {}
    static constexpr unsigned ValueBits = StateBits;
    static constexpr unsigned KindBits = 3;
    static_assert(ValueBits >= RuleBits, "Value must be able to store RuleID");
    static_assert(KindBits + ValueBits <= 16,
                  "Must be able to store kind and value efficiently");
    uint16_t K : KindBits;
    // Either StateID or RuleID, depending on the Kind.
    uint16_t Value : ValueBits;
  };

  // Returns all available actions for the given state on a terminal.
  // Expected to be called by LR parsers.
  llvm::ArrayRef<Action> getActions(StateID State, SymbolID Terminal) const;
  // Returns the state after we reduce a nonterminal.
  // Expected to be called by LR parsers.
  StateID getGoToState(StateID State, SymbolID Nonterminal) const;

  // Looks up available actions.
  // Returns empty if no available actions in the table.
  llvm::ArrayRef<Action> find(StateID State, SymbolID Symbol) const;

  // Returns the state from which the LR parser should start to parse the input
  // tokens as the given StartSymbol.
  //
  // In LR parsing, the start state of `translation-unit` corresponds to
  // `_ := • translation-unit`.
  //
  // Each start state responds to **a** single grammar rule like `_ := start`.
  // REQUIRE: The given StartSymbol must exist in the grammar (in a form of
  //          `_ := start`).
  StateID getStartState(SymbolID StartSymbol) const;

  size_t bytes() const {
    return sizeof(*this) + Actions.capacity() * sizeof(Action) +
           Symbols.capacity() * sizeof(SymbolID) +
           StateOffset.capacity() * sizeof(uint32_t);
  }

  std::string dumpStatistics() const;
  std::string dumpForTests(const Grammar &G) const;

  // Build a SLR(1) parsing table.
  static LRTable buildSLR(const Grammar &G);

  class Builder;
  // Represents an entry in the table, used for building the LRTable.
  struct Entry {
    StateID State;
    SymbolID Symbol;
    Action Act;
  };
  // Build a specifid table for testing purposes.
  static LRTable buildForTests(const GrammarTable &, llvm::ArrayRef<Entry>);

private:
  // Conceptually the LR table is a multimap from (State, SymbolID) => Action.
  // Our physical representation is quite different for compactness.

  // Index is StateID, value is the offset into Symbols/Actions
  // where the entries for this state begin.
  // Give a state id, the corresponding half-open range of Symbols/Actions is
  // [StateOffset[id], StateOffset[id+1]).
  std::vector<uint32_t> StateOffset;
  // Parallel to Actions, the value is SymbolID (columns of the matrix).
  // Grouped by the StateID, and only subranges are sorted.
  std::vector<SymbolID> Symbols;
  // A flat list of available actions, sorted by (State, SymbolID).
  std::vector<Action> Actions;
  // A sorted table, storing the start state for each target parsing symbol.
  std::vector<std::pair<SymbolID, StateID>> StartStates;
};
llvm::raw_ostream &operator<<(llvm::raw_ostream &, const LRTable::Action &);

} // namespace pseudo
} // namespace clang

#endif // CLANG_PSEUDO_GRAMMAR_LRTABLE_H
