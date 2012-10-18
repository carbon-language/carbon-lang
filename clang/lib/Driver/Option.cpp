//===--- Option.cpp - Abstract Driver Options -----------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "clang/Driver/Option.h"

#include "clang/Driver/Arg.h"
#include "clang/Driver/ArgList.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/ErrorHandling.h"
#include <cassert>
#include <algorithm>
using namespace clang::driver;

Option::Option(const OptTable::Info *info, const OptTable *owner)
  : Info(info), Owner(owner) {

  // Multi-level aliases are not supported, and alias options cannot
  // have groups. This just simplifies option tracking, it is not an
  // inherent limitation.
  assert((!getAlias() || (!getAlias()->getAlias() && !getGroup())) &&
         "Multi-level aliases and aliases with groups are unsupported.");
}

Option::~Option() {
}

void Option::dump() const {
  llvm::errs() << "<";
  switch (getKind()) {
#define P(N) case N: llvm::errs() << #N; break
    P(GroupClass);
    P(InputClass);
    P(UnknownClass);
    P(FlagClass);
    P(JoinedClass);
    P(SeparateClass);
    P(CommaJoinedClass);
    P(MultiArgClass);
    P(JoinedOrSeparateClass);
    P(JoinedAndSeparateClass);
#undef P
  }

  llvm::errs() << " Name:\"" << getName() << '"';

  const Option *Group = getGroup();
  if (Group) {
    llvm::errs() << " Group:";
    Group->dump();
  }

  const Option *Alias = getAlias();
  if (Alias) {
    llvm::errs() << " Alias:";
    Alias->dump();
  }

  if (getKind() == MultiArgClass)
    llvm::errs() << " NumArgs:" << getNumArgs();

  llvm::errs() << ">\n";
}

bool Option::matches(OptSpecifier Opt) const {
  // Aliases are never considered in matching, look through them.
  const Option *Alias = getAlias();
  if (Alias)
    return Alias->matches(Opt);

  // Check exact match.
  if (getID() == Opt.getID())
    return true;

  const Option *Group = getGroup();
  if (Group)
    return Group->matches(Opt);
  return false;
}

Arg *Option::accept(const ArgList &Args, unsigned &Index) const {
  switch (getKind()) {
  case FlagClass:
    if (getName().size() != strlen(Args.getArgString(Index)))
      return 0;

    return new Arg(getUnaliasedOption(), Index++);
  case JoinedClass: {
    const char *Value = Args.getArgString(Index) + getName().size();
    return new Arg(getUnaliasedOption(), Index++, Value);
  }
  case CommaJoinedClass: {
    // Always matches.
    const char *Str = Args.getArgString(Index) + getName().size();
    Arg *A = new Arg(getUnaliasedOption(), Index++);

    // Parse out the comma separated values.
    const char *Prev = Str;
    for (;; ++Str) {
      char c = *Str;

      if (!c || c == ',') {
        if (Prev != Str) {
          char *Value = new char[Str - Prev + 1];
          memcpy(Value, Prev, Str - Prev);
          Value[Str - Prev] = '\0';
          A->getValues().push_back(Value);
        }

        if (!c)
          break;

        Prev = Str + 1;
      }
    }
    A->setOwnsValues(true);

    return A;
  }
  case SeparateClass:
    // Matches iff this is an exact match.
    // FIXME: Avoid strlen.
    if (getName().size() != strlen(Args.getArgString(Index)))
      return 0;

    Index += 2;
    if (Index > Args.getNumInputArgStrings())
      return 0;

    return new Arg(getUnaliasedOption(),
                   Index - 2, Args.getArgString(Index - 1));
  case MultiArgClass: {
    // Matches iff this is an exact match.
    // FIXME: Avoid strlen.
    if (getName().size() != strlen(Args.getArgString(Index)))
      return 0;

    Index += 1 + getNumArgs();
    if (Index > Args.getNumInputArgStrings())
      return 0;

    Arg *A = new Arg(getUnaliasedOption(), Index - 1 - getNumArgs(),
                      Args.getArgString(Index - getNumArgs()));
    for (unsigned i = 1; i != getNumArgs(); ++i)
      A->getValues().push_back(Args.getArgString(Index - getNumArgs() + i));
    return A;
  }
  case JoinedOrSeparateClass: {
    // If this is not an exact match, it is a joined arg.
    // FIXME: Avoid strlen.
    if (getName().size() != strlen(Args.getArgString(Index))) {
      const char *Value = Args.getArgString(Index) + getName().size();
      return new Arg(this, Index++, Value);
    }

    // Otherwise it must be separate.
    Index += 2;
    if (Index > Args.getNumInputArgStrings())
      return 0;

    return new Arg(getUnaliasedOption(),
                   Index - 2, Args.getArgString(Index - 1));
  }
  case JoinedAndSeparateClass:
    // Always matches.
    Index += 2;
    if (Index > Args.getNumInputArgStrings())
      return 0;

    return new Arg(getUnaliasedOption(), Index - 2,
                   Args.getArgString(Index-2)+getName().size(),
                   Args.getArgString(Index-1));
  default:
    llvm_unreachable("Invalid option kind!");
  }
}
