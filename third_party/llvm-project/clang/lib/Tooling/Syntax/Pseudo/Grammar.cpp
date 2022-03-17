//===--- Grammar.cpp - Grammar for clang pseudo parser  ----------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Tooling/Syntax/Pseudo/Grammar.h"
#include "clang/Basic/TokenKinds.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"

namespace clang {
namespace syntax {
namespace pseudo {

Rule::Rule(SymbolID Target, llvm::ArrayRef<SymbolID> Sequence)
    : Target(Target), Size(static_cast<uint8_t>(Sequence.size())) {
  assert(Sequence.size() <= Rule::MaxElements);
  llvm::copy(Sequence, this->Sequence);
}

Grammar::Grammar(std::unique_ptr<GrammarTable> Table) : T(std::move(Table)) {
  // start symbol is named _, binary search it.
  auto It = llvm::partition_point(
      T->Nonterminals,
      [](const GrammarTable::Nonterminal &X) { return X.Name < "_"; });
  assert(It != T->Nonterminals.end() && It->Name == "_" &&
         "symbol _ must exist in the grammar!");
  StartSymbol = It - T->Nonterminals.begin();
}

llvm::ArrayRef<Rule> Grammar::rulesFor(SymbolID SID) const {
  assert(isNonterminal(SID));
  const auto &R = T->Nonterminals[SID].RuleRange;
  assert(R.end <= T->Rules.size());
  return llvm::makeArrayRef(&T->Rules[R.start], R.end - R.start);
}

const Rule &Grammar::lookupRule(RuleID RID) const {
  assert(RID < T->Rules.size());
  return T->Rules[RID];
}

llvm::StringRef Grammar::symbolName(SymbolID SID) const {
  if (isToken(SID))
    return T->Terminals[symbolToToken(SID)];
  return T->Nonterminals[SID].Name;
}

std::string Grammar::dumpRule(RuleID RID) const {
  std::string Result;
  llvm::raw_string_ostream OS(Result);
  const Rule &R = T->Rules[RID];
  OS << symbolName(R.Target) << " :=";
  for (SymbolID SID : R.seq())
    OS << " " << symbolName(SID);
  return Result;
}

std::string Grammar::dumpRules(SymbolID SID) const {
  assert(isNonterminal(SID));
  std::string Result;
  const auto &Range = T->Nonterminals[SID].RuleRange;
  for (RuleID RID = Range.start; RID < Range.end; ++RID)
    Result.append(dumpRule(RID)).push_back('\n');
  return Result;
}

std::string Grammar::dump() const {
  std::string Result;
  llvm::raw_string_ostream OS(Result);
  OS << "Nonterminals:\n";
  for (SymbolID SID = 0; SID < T->Nonterminals.size(); ++SID)
    OS << llvm::formatv("  {0} {1}\n", SID, symbolName(SID));
  OS << "Rules:\n";
  for (RuleID RID = 0; RID < T->Rules.size(); ++RID)
    OS << llvm::formatv("  {0} {1}\n", RID, dumpRule(RID));
  return OS.str();
}

std::vector<llvm::DenseSet<SymbolID>> firstSets(const Grammar &G) {
  std::vector<llvm::DenseSet<SymbolID>> FirstSets(
      G.table().Nonterminals.size());
  auto ExpandFirstSet = [&FirstSets](SymbolID Target, SymbolID First) {
    assert(isNonterminal(Target));
    if (isToken(First))
      return FirstSets[Target].insert(First).second;
    bool Changed = false;
    for (SymbolID SID : FirstSets[First])
      Changed |= FirstSets[Target].insert(SID).second;
    return Changed;
  };

  // A rule S := T ... implies elements in FIRST(S):
  //  - if T is a terminal, FIRST(S) contains T
  //  - if T is a nonterminal, FIRST(S) contains FIRST(T)
  // Since FIRST(T) may not have been fully computed yet, FIRST(S) itself may
  // end up being incomplete.
  // We iterate until we hit a fixed point.
  // (This isn't particularly efficient, but table building isn't on the
  // critical path).
  bool Changed = true;
  while (Changed) {
    Changed = false;
    for (const auto &R : G.table().Rules)
      // We only need to consider the first element because symbols are
      // non-nullable.
      Changed |= ExpandFirstSet(R.Target, R.seq().front());
  }
  return FirstSets;
}

std::vector<llvm::DenseSet<SymbolID>> followSets(const Grammar &G) {
  auto FirstSets = firstSets(G);
  std::vector<llvm::DenseSet<SymbolID>> FollowSets(
      G.table().Nonterminals.size());
  // Expand the follow set of a non-terminal symbol Y by adding all from the
  // given symbol set.
  auto ExpandFollowSet = [&FollowSets](SymbolID Y,
                                       const llvm::DenseSet<SymbolID> &ToAdd) {
    assert(isNonterminal(Y));
    bool Changed = false;
    for (SymbolID F : ToAdd)
      Changed |= FollowSets[Y].insert(F).second;
    return Changed;
  };
  // Follow sets is computed based on the following 3 rules, the computation
  // is completed at a fixed point where there is no more new symbols can be
  // added to any of the follow sets.
  //
  // Rule 1: add endmarker to the FOLLOW(S), where S is the start symbol.
  FollowSets[G.startSymbol()].insert(tokenSymbol(tok::eof));
  bool Changed = true;
  while (Changed) {
    Changed = false;
    for (const auto &R : G.table().Rules) {
      // Rule 2: for a rule X := ... Y Z, we add all symbols from FIRST(Z) to
      // FOLLOW(Y).
      for (size_t i = 0; i + 1 < R.seq().size(); ++i) {
        if (isToken(R.seq()[i]))
          continue;
        // We only need to consider the next symbol because symbols are
        // non-nullable.
        SymbolID Next = R.seq()[i + 1];
        if (isToken(Next))
          // First set for a terminal is itself.
          Changed |= ExpandFollowSet(R.seq()[i], {Next});
        else
          Changed |= ExpandFollowSet(R.seq()[i], FirstSets[Next]);
      }
      // Rule 3: for a rule X := ... Z, we add all symbols from FOLLOW(X) to
      // FOLLOW(Z).
      SymbolID Z = R.seq().back();
      if (isNonterminal(Z))
        Changed |= ExpandFollowSet(Z, FollowSets[R.Target]);
    }
  }
  return FollowSets;
}

static llvm::ArrayRef<std::string> getTerminalNames() {
  static const std::vector<std::string> *TerminalNames = []() {
    static std::vector<std::string> TerminalNames;
    TerminalNames.reserve(NumTerminals);
    for (unsigned I = 0; I < NumTerminals; ++I) {
      tok::TokenKind K = static_cast<tok::TokenKind>(I);
      if (const auto *Punc = tok::getPunctuatorSpelling(K))
        TerminalNames.push_back(Punc);
      else
        TerminalNames.push_back(llvm::StringRef(tok::getTokenName(K)).upper());
    }
    return &TerminalNames;
  }();
  return *TerminalNames;
}
GrammarTable::GrammarTable() : Terminals(getTerminalNames()) {}

} // namespace pseudo
} // namespace syntax
} // namespace clang
