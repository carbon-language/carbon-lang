//===--- Option.cpp - Abstract Driver Options ---------------------------*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "clang/Driver/Option.h"
#include <cassert>
using namespace clang;
using namespace clang::driver;

Option::Option(OptionClass _Kind, const char *_Name,
               OptionGroup *_Group, Option *_Alias) 
  : Kind(_Kind), Name(_Name), Group(_Group), Alias(_Alias) {

  // Multi-level aliases are not supported, and alias options cannot
  // have groups. This just simplifies option tracking, it is not an
  // inherent limitation.
  assert((!Alias || (!Alias->Alias && !Group)) &&
         "Multi-level aliases and aliases with groups are unsupported.");    
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

OptionGroup::OptionGroup(const char *Name, OptionGroup *Group)
  : Option(Option::GroupOption, Name, Group, 0) {
}

InputOption::InputOption()
  : Option(Option::InputOption, "<input>", 0, 0) {
}

UnknownOption::UnknownOption()
  : Option(Option::UnknownOption, "<unknown>", 0, 0) {
}

FlagOption::FlagOption(const char *Name, OptionGroup *Group, Option *Alias)
  : Option(Option::FlagOption, Name, Group, Alias) {
}


JoinedOption::JoinedOption(const char *Name, OptionGroup *Group, Option *Alias)
  : Option(Option::JoinedOption, Name, Group, Alias) {
}

CommaJoinedOption::CommaJoinedOption(const char *Name, OptionGroup *Group, 
                                     Option *Alias)
  : Option(Option::CommaJoinedOption, Name, Group, Alias) {
}

SeparateOption::SeparateOption(const char *Name, OptionGroup *Group, 
                               Option *Alias)
  : Option(Option::SeparateOption, Name, Group, Alias) {
}

MultiArgOption::MultiArgOption(const char *Name, OptionGroup *Group, 
                               Option *Alias, unsigned _NumArgs)
  : Option(Option::MultiArgOption, Name, Group, Alias), NumArgs(_NumArgs) {
}

JoinedOrSeparateOption::JoinedOrSeparateOption(const char *Name, 
                                               OptionGroup *Group, 
                                               Option *Alias)
  : Option(Option::JoinedOrSeparateOption, Name, Group, Alias) {
}

JoinedAndSeparateOption::JoinedAndSeparateOption(const char *Name, 
                                                 OptionGroup *Group, 
                                                 Option *Alias)
  : Option(Option::JoinedAndSeparateOption, Name, Group, Alias) {
}


