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
using namespace clang;
using namespace clang::driver;

Option::Option(OptionClass _Kind, options::ID _ID, const char *_Name,
               const OptionGroup *_Group, const Option *_Alias) 
  : Kind(_Kind), ID(_ID), Name(_Name), Group(_Group), Alias(_Alias),
    Unsupported(false), LinkerInput(false), NoOptAsInput(false),
    ForceSeparateRender(false), ForceJoinedRender(false)
{

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

  llvm::errs().flush(); // FIXME
}

bool Option::matches(const Option *Opt) const {
  // Aliases are never considered in matching.
  if (Opt->getAlias())
    return matches(Opt->getAlias());
  if (Alias)
    return Alias->matches(Opt);
  
  if (this == Opt)
    return true;
  
  if (Group)
    return Group->matches(Opt);
  return false;
}

OptionGroup::OptionGroup(options::ID ID, const char *Name, 
                         const OptionGroup *Group)
  : Option(Option::GroupClass, ID, Name, Group, 0) {
}

Arg *OptionGroup::accept(const ArgList &Args, unsigned &Index) const {
  assert(0 && "FIXME");
  return 0;
}

InputOption::InputOption()
  : Option(Option::InputClass, options::InputOpt, "<input>", 0, 0) {
}

Arg *InputOption::accept(const ArgList &Args, unsigned &Index) const {
  assert(0 && "FIXME");
  return 0;
}

UnknownOption::UnknownOption()
  : Option(Option::UnknownClass, options::UnknownOpt, "<unknown>", 0, 0) {
}

Arg *UnknownOption::accept(const ArgList &Args, unsigned &Index) const {
  assert(0 && "FIXME");
  return 0;
}

FlagOption::FlagOption(options::ID ID, const char *Name, 
                       const OptionGroup *Group, const Option *Alias)
  : Option(Option::FlagClass, ID, Name, Group, Alias) {
}

Arg *FlagOption::accept(const ArgList &Args, unsigned &Index) const {
  // Matches iff this is an exact match.  
  // FIXME: Avoid strlen.
  if (strlen(getName()) != strlen(Args.getArgString(Index)))
    return 0;

  return new PositionalArg(this, Index++);
}

JoinedOption::JoinedOption(options::ID ID, const char *Name, 
                           const OptionGroup *Group, const Option *Alias)
  : Option(Option::JoinedClass, ID, Name, Group, Alias) {
}

Arg *JoinedOption::accept(const ArgList &Args, unsigned &Index) const {
  // Always matches.
  return new JoinedArg(this, Index++);
}

CommaJoinedOption::CommaJoinedOption(options::ID ID, const char *Name, 
                                     const OptionGroup *Group, 
                                     const Option *Alias)
  : Option(Option::CommaJoinedClass, ID, Name, Group, Alias) {
}

Arg *CommaJoinedOption::accept(const ArgList &Args, unsigned &Index) const {
  // Always matches. We count the commas now so we can answer
  // getNumValues easily.
  
  // Get the suffix string.
  // FIXME: Avoid strlen, and move to helper method?
  const char *Suffix = Args.getArgString(Index) + strlen(getName());
  const char *SuffixEnd = Suffix + strlen(Suffix);
  
  // Degenerate case, exact match has no values.
  if (Suffix == SuffixEnd)
    return new CommaJoinedArg(this, Index++, 0);

  return new CommaJoinedArg(this, Index++, 
                            std::count(Suffix, SuffixEnd, ',') + 1);
}

SeparateOption::SeparateOption(options::ID ID, const char *Name, 
                               const OptionGroup *Group, const Option *Alias)
  : Option(Option::SeparateClass, ID, Name, Group, Alias) {
}

Arg *SeparateOption::accept(const ArgList &Args, unsigned &Index) const {
  // Matches iff this is an exact match.  
  // FIXME: Avoid strlen.
  if (strlen(getName()) != strlen(Args.getArgString(Index)))
    return 0;

  // FIXME: Missing argument error.
  Index += 2;
  return new SeparateArg(this, Index - 2, 1);
}

MultiArgOption::MultiArgOption(options::ID ID, const char *Name, 
                               const OptionGroup *Group, const Option *Alias, 
                               unsigned _NumArgs)
  : Option(Option::MultiArgClass, ID, Name, Group, Alias), NumArgs(_NumArgs) {
}

Arg *MultiArgOption::accept(const ArgList &Args, unsigned &Index) const {
  // Matches iff this is an exact match.  
  // FIXME: Avoid strlen.
  if (strlen(getName()) != strlen(Args.getArgString(Index)))
    return 0;

  // FIXME: Missing argument error.
  Index += 1 + NumArgs;
  return new SeparateArg(this, Index - 1 - NumArgs, NumArgs);
}

JoinedOrSeparateOption::JoinedOrSeparateOption(options::ID ID, const char *Name,
                                               const OptionGroup *Group, 
                                               const Option *Alias)
  : Option(Option::JoinedOrSeparateClass, ID, Name, Group, Alias) {
}

Arg *JoinedOrSeparateOption::accept(const ArgList &Args, unsigned &Index) const {
  // If this is not an exact match, it is a joined arg.
  // FIXME: Avoid strlen.
  if (strlen(getName()) != strlen(Args.getArgString(Index)))
    return new JoinedArg(this, Index++);

  // Otherwise it must be separate.
  // FIXME: Missing argument error.
  Index += 2;
  return new SeparateArg(this, Index - 2, 1);  
}

JoinedAndSeparateOption::JoinedAndSeparateOption(options::ID ID,
                                                 const char *Name, 
                                                 const OptionGroup *Group, 
                                                 const Option *Alias)
  : Option(Option::JoinedAndSeparateClass, ID, Name, Group, Alias) {
}

Arg *JoinedAndSeparateOption::accept(const ArgList &Args, unsigned &Index) const {
  // Always matches.

  // FIXME: Missing argument error.
  Index += 2;
  return new JoinedAndSeparateArg(this, Index - 2);
}

