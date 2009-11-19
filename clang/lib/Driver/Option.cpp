//===--- Option.cpp - Abstract Driver Options ---------------------------*-===//
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
#include <cassert>
#include <algorithm>
using namespace clang::driver;

Option::Option(OptionClass _Kind, OptSpecifier _ID, const char *_Name,
               const OptionGroup *_Group, const Option *_Alias)
  : Kind(_Kind), ID(_ID.getID()), Name(_Name), Group(_Group), Alias(_Alias),
    Unsupported(false), LinkerInput(false), NoOptAsInput(false),
    ForceSeparateRender(false), ForceJoinedRender(false),
    DriverOption(false), NoArgumentUnused(false) {

  // Multi-level aliases are not supported, and alias options cannot
  // have groups. This just simplifies option tracking, it is not an
  // inherent limitation.
  assert((!Alias || (!Alias->Alias && !Group)) &&
         "Multi-level aliases and aliases with groups are unsupported.");
}

Option::~Option() {
}

void Option::dump() const {
  llvm::errs() << "<";
  switch (Kind) {
  default:
    assert(0 && "Invalid kind");
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

Arg *OptionGroup::accept(const InputArgList &Args, unsigned &Index) const {
  assert(0 && "accept() should never be called on an OptionGroup");
  return 0;
}

InputOption::InputOption(OptSpecifier ID)
  : Option(Option::InputClass, ID, "<input>", 0, 0) {
}

Arg *InputOption::accept(const InputArgList &Args, unsigned &Index) const {
  assert(0 && "accept() should never be called on an InputOption");
  return 0;
}

UnknownOption::UnknownOption(OptSpecifier ID)
  : Option(Option::UnknownClass, ID, "<unknown>", 0, 0) {
}

Arg *UnknownOption::accept(const InputArgList &Args, unsigned &Index) const {
  assert(0 && "accept() should never be called on an UnknownOption");
  return 0;
}

FlagOption::FlagOption(OptSpecifier ID, const char *Name,
                       const OptionGroup *Group, const Option *Alias)
  : Option(Option::FlagClass, ID, Name, Group, Alias) {
}

Arg *FlagOption::accept(const InputArgList &Args, unsigned &Index) const {
  // Matches iff this is an exact match.
  // FIXME: Avoid strlen.
  if (strlen(getName()) != strlen(Args.getArgString(Index)))
    return 0;

  return new FlagArg(this, Index++);
}

JoinedOption::JoinedOption(OptSpecifier ID, const char *Name,
                           const OptionGroup *Group, const Option *Alias)
  : Option(Option::JoinedClass, ID, Name, Group, Alias) {
}

Arg *JoinedOption::accept(const InputArgList &Args, unsigned &Index) const {
  // Always matches.
  return new JoinedArg(this, Index++);
}

CommaJoinedOption::CommaJoinedOption(OptSpecifier ID, const char *Name,
                                     const OptionGroup *Group,
                                     const Option *Alias)
  : Option(Option::CommaJoinedClass, ID, Name, Group, Alias) {
}

Arg *CommaJoinedOption::accept(const InputArgList &Args,
                               unsigned &Index) const {
  // Always matches. We count the commas now so we can answer
  // getNumValues easily.

  // Get the suffix string.
  // FIXME: Avoid strlen, and move to helper method?
  const char *Suffix = Args.getArgString(Index) + strlen(getName());
  return new CommaJoinedArg(this, Index++, Suffix);
}

SeparateOption::SeparateOption(OptSpecifier ID, const char *Name,
                               const OptionGroup *Group, const Option *Alias)
  : Option(Option::SeparateClass, ID, Name, Group, Alias) {
}

Arg *SeparateOption::accept(const InputArgList &Args, unsigned &Index) const {
  // Matches iff this is an exact match.
  // FIXME: Avoid strlen.
  if (strlen(getName()) != strlen(Args.getArgString(Index)))
    return 0;

  Index += 2;
  if (Index > Args.getNumInputArgStrings())
    return 0;

  return new SeparateArg(this, Index - 2, 1);
}

MultiArgOption::MultiArgOption(OptSpecifier ID, const char *Name,
                               const OptionGroup *Group, const Option *Alias,
                               unsigned _NumArgs)
  : Option(Option::MultiArgClass, ID, Name, Group, Alias), NumArgs(_NumArgs) {
  assert(NumArgs > 1  && "Invalid MultiArgOption!");
}

Arg *MultiArgOption::accept(const InputArgList &Args, unsigned &Index) const {
  // Matches iff this is an exact match.
  // FIXME: Avoid strlen.
  if (strlen(getName()) != strlen(Args.getArgString(Index)))
    return 0;

  Index += 1 + NumArgs;
  if (Index > Args.getNumInputArgStrings())
    return 0;

  return new SeparateArg(this, Index - 1 - NumArgs, NumArgs);
}

JoinedOrSeparateOption::JoinedOrSeparateOption(OptSpecifier ID, const char *Name,
                                               const OptionGroup *Group,
                                               const Option *Alias)
  : Option(Option::JoinedOrSeparateClass, ID, Name, Group, Alias) {
}

Arg *JoinedOrSeparateOption::accept(const InputArgList &Args,
                                    unsigned &Index) const {
  // If this is not an exact match, it is a joined arg.
  // FIXME: Avoid strlen.
  if (strlen(getName()) != strlen(Args.getArgString(Index)))
    return new JoinedArg(this, Index++);

  // Otherwise it must be separate.
  Index += 2;
  if (Index > Args.getNumInputArgStrings())
    return 0;

  return new SeparateArg(this, Index - 2, 1);
}

JoinedAndSeparateOption::JoinedAndSeparateOption(OptSpecifier ID,
                                                 const char *Name,
                                                 const OptionGroup *Group,
                                                 const Option *Alias)
  : Option(Option::JoinedAndSeparateClass, ID, Name, Group, Alias) {
}

Arg *JoinedAndSeparateOption::accept(const InputArgList &Args,
                                     unsigned &Index) const {
  // Always matches.

  Index += 2;
  if (Index > Args.getNumInputArgStrings())
    return 0;

  return new JoinedAndSeparateArg(this, Index - 2);
}

