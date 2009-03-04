//===--- Option.cpp - Abstract Driver Options ---------------------------*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "clang/Driver/Option.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>
using namespace clang;
using namespace clang::driver;

Option::Option(OptionClass _Kind, const char *_Name,
               const OptionGroup *_Group, const Option *_Alias) 
  : Kind(_Kind), Name(_Name), Group(_Group), Alias(_Alias),
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

OptionGroup::OptionGroup(const char *Name, const OptionGroup *Group)
  : Option(Option::GroupClass, Name, Group, 0) {
}

Arg *OptionGroup::accept(const ArgList &Args, unsigned &Index) const {
  assert(0 && "FIXME");
  return 0;
}

InputOption::InputOption()
  : Option(Option::InputClass, "<input>", 0, 0) {
}

Arg *InputOption::accept(const ArgList &Args, unsigned &Index) const {
  assert(0 && "FIXME");
  return 0;
}

UnknownOption::UnknownOption()
  : Option(Option::UnknownClass, "<unknown>", 0, 0) {
}

Arg *UnknownOption::accept(const ArgList &Args, unsigned &Index) const {
  assert(0 && "FIXME");
  return 0;
}

FlagOption::FlagOption(const char *Name, const OptionGroup *Group, 
                       const Option *Alias)
  : Option(Option::FlagClass, Name, Group, Alias) {
}

Arg *FlagOption::accept(const ArgList &Args, unsigned &Index) const {
  assert(0 && "FIXME");
  return 0;
}

JoinedOption::JoinedOption(const char *Name, const OptionGroup *Group, 
                           const Option *Alias)
  : Option(Option::JoinedClass, Name, Group, Alias) {
}

Arg *JoinedOption::accept(const ArgList &Args, unsigned &Index) const {
  assert(0 && "FIXME");
  return 0;
}

CommaJoinedOption::CommaJoinedOption(const char *Name, const OptionGroup *Group, 
                                     const Option *Alias)
  : Option(Option::CommaJoinedClass, Name, Group, Alias) {
}

Arg *CommaJoinedOption::accept(const ArgList &Args, unsigned &Index) const {
  assert(0 && "FIXME");
  return 0;
}

SeparateOption::SeparateOption(const char *Name, const OptionGroup *Group, 
                               const Option *Alias)
  : Option(Option::SeparateClass, Name, Group, Alias) {
}

Arg *SeparateOption::accept(const ArgList &Args, unsigned &Index) const {
  assert(0 && "FIXME");
  return 0;
}

MultiArgOption::MultiArgOption(const char *Name, const OptionGroup *Group, 
                               const Option *Alias, unsigned _NumArgs)
  : Option(Option::MultiArgClass, Name, Group, Alias), NumArgs(_NumArgs) {
}

Arg *MultiArgOption::accept(const ArgList &Args, unsigned &Index) const {
  assert(0 && "FIXME");
  return 0;
}

JoinedOrSeparateOption::JoinedOrSeparateOption(const char *Name, 
                                               const OptionGroup *Group, 
                                               const Option *Alias)
  : Option(Option::JoinedOrSeparateClass, Name, Group, Alias) {
}

Arg *JoinedOrSeparateOption::accept(const ArgList &Args, unsigned &Index) const {
  assert(0 && "FIXME");
  return 0;
}

JoinedAndSeparateOption::JoinedAndSeparateOption(const char *Name, 
                                                 const OptionGroup *Group, 
                                                 const Option *Alias)
  : Option(Option::JoinedAndSeparateClass, Name, Group, Alias) {
}

Arg *JoinedAndSeparateOption::accept(const ArgList &Args, unsigned &Index) const {
  assert(0 && "FIXME");
  return 0;
}

