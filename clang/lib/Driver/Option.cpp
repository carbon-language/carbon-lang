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

Option::Option(OptionClass _Kind, OptSpecifier _ID, const char *_Name,
               const OptionGroup *_Group, const Option *_Alias)
  : Kind(_Kind), ID(_ID.getID()), Name(_Name), Group(_Group), Alias(_Alias),
    Unsupported(false), LinkerInput(false), NoOptAsInput(false),
    DriverOption(false), NoArgumentUnused(false), NoForward(false) {

  // Multi-level aliases are not supported, and alias options cannot
  // have groups. This just simplifies option tracking, it is not an
  // inherent limitation.
  assert((!Alias || (!Alias->Alias && !Group)) &&
         "Multi-level aliases and aliases with groups are unsupported.");

  // Initialize rendering options based on the class.
  switch (Kind) {
  case GroupClass:
  case InputClass:
  case UnknownClass:
    RenderStyle = RenderValuesStyle;
    break;

  case JoinedClass:
  case JoinedAndSeparateClass:
    RenderStyle = RenderJoinedStyle;
    break;

  case CommaJoinedClass:
    RenderStyle = RenderCommaJoinedStyle;
    break;

  case FlagClass:
  case SeparateClass:
  case MultiArgClass:
  case JoinedOrSeparateClass:
    RenderStyle = RenderSeparateStyle;
    break;
  }
}

Option::~Option() {
}

void Option::dump() const {
  llvm::errs() << "<";
  switch (Kind) {
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

  llvm::errs() << " Name:\"" << Name << '"';

  if (Group) {
    llvm::errs() << " Group:";
    Group->dump();
  }

  if (Alias) {
    llvm::errs() << " Alias:";
    Alias->dump();
  }

  if (const MultiArgOption *MOA = dyn_cast<MultiArgOption>(this))
    llvm::errs() << " NumArgs:" << MOA->getNumArgs();

  llvm::errs() << ">\n";
}

bool Option::matches(OptSpecifier Opt) const {
  // Aliases are never considered in matching, look through them.
  if (Alias)
    return Alias->matches(Opt);

  // Check exact match.
  if (ID == Opt)
    return true;

  if (Group)
    return Group->matches(Opt);
  return false;
}

OptionGroup::OptionGroup(OptSpecifier ID, const char *Name,
                         const OptionGroup *Group)
  : Option(Option::GroupClass, ID, Name, Group, 0) {
}

Arg *OptionGroup::accept(const ArgList &Args, unsigned &Index) const {
  llvm_unreachable("accept() should never be called on an OptionGroup");
}

InputOption::InputOption(OptSpecifier ID)
  : Option(Option::InputClass, ID, "<input>", 0, 0) {
}

Arg *InputOption::accept(const ArgList &Args, unsigned &Index) const {
  llvm_unreachable("accept() should never be called on an InputOption");
}

UnknownOption::UnknownOption(OptSpecifier ID)
  : Option(Option::UnknownClass, ID, "<unknown>", 0, 0) {
}

Arg *UnknownOption::accept(const ArgList &Args, unsigned &Index) const {
  llvm_unreachable("accept() should never be called on an UnknownOption");
}

FlagOption::FlagOption(OptSpecifier ID, const char *Name,
                       const OptionGroup *Group, const Option *Alias)
  : Option(Option::FlagClass, ID, Name, Group, Alias) {
}

Arg *FlagOption::accept(const ArgList &Args, unsigned &Index) const {
  // Matches iff this is an exact match.
  // FIXME: Avoid strlen.
  if (getName().size() != strlen(Args.getArgString(Index)))
    return 0;

  return new Arg(getUnaliasedOption(), Index++);
}

JoinedOption::JoinedOption(OptSpecifier ID, const char *Name,
                           const OptionGroup *Group, const Option *Alias)
  : Option(Option::JoinedClass, ID, Name, Group, Alias) {
}

Arg *JoinedOption::accept(const ArgList &Args, unsigned &Index) const {
  // Always matches.
  const char *Value = Args.getArgString(Index) + getName().size();
  return new Arg(getUnaliasedOption(), Index++, Value);
}

CommaJoinedOption::CommaJoinedOption(OptSpecifier ID, const char *Name,
                                     const OptionGroup *Group,
                                     const Option *Alias)
  : Option(Option::CommaJoinedClass, ID, Name, Group, Alias) {
}

Arg *CommaJoinedOption::accept(const ArgList &Args,
                               unsigned &Index) const {
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

SeparateOption::SeparateOption(OptSpecifier ID, const char *Name,
                               const OptionGroup *Group, const Option *Alias)
  : Option(Option::SeparateClass, ID, Name, Group, Alias) {
}

Arg *SeparateOption::accept(const ArgList &Args, unsigned &Index) const {
  // Matches iff this is an exact match.
  // FIXME: Avoid strlen.
  if (getName().size() != strlen(Args.getArgString(Index)))
    return 0;

  Index += 2;
  if (Index > Args.getNumInputArgStrings())
    return 0;

  return new Arg(getUnaliasedOption(), Index - 2, Args.getArgString(Index - 1));
}

MultiArgOption::MultiArgOption(OptSpecifier ID, const char *Name,
                               const OptionGroup *Group, const Option *Alias,
                               unsigned _NumArgs)
  : Option(Option::MultiArgClass, ID, Name, Group, Alias), NumArgs(_NumArgs) {
  assert(NumArgs > 1  && "Invalid MultiArgOption!");
}

Arg *MultiArgOption::accept(const ArgList &Args, unsigned &Index) const {
  // Matches iff this is an exact match.
  // FIXME: Avoid strlen.
  if (getName().size() != strlen(Args.getArgString(Index)))
    return 0;

  Index += 1 + NumArgs;
  if (Index > Args.getNumInputArgStrings())
    return 0;

  Arg *A = new Arg(getUnaliasedOption(), Index - 1 - NumArgs,
                   Args.getArgString(Index - NumArgs));
  for (unsigned i = 1; i != NumArgs; ++i)
    A->getValues().push_back(Args.getArgString(Index - NumArgs + i));
  return A;
}

JoinedOrSeparateOption::JoinedOrSeparateOption(OptSpecifier ID,
                                               const char *Name,
                                               const OptionGroup *Group,
                                               const Option *Alias)
  : Option(Option::JoinedOrSeparateClass, ID, Name, Group, Alias) {
}

Arg *JoinedOrSeparateOption::accept(const ArgList &Args,
                                    unsigned &Index) const {
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

  return new Arg(getUnaliasedOption(), Index - 2, Args.getArgString(Index - 1));
}

JoinedAndSeparateOption::JoinedAndSeparateOption(OptSpecifier ID,
                                                 const char *Name,
                                                 const OptionGroup *Group,
                                                 const Option *Alias)
  : Option(Option::JoinedAndSeparateClass, ID, Name, Group, Alias) {
}

Arg *JoinedAndSeparateOption::accept(const ArgList &Args,
                                     unsigned &Index) const {
  // Always matches.

  Index += 2;
  if (Index > Args.getNumInputArgStrings())
    return 0;

  return new Arg(getUnaliasedOption(), Index - 2,
                 Args.getArgString(Index-2)+getName().size(),
                 Args.getArgString(Index-1));
}
