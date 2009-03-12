//===--- ArgList.cpp - Argument List Management -------------------------*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "clang/Driver/ArgList.h"
#include "clang/Driver/Arg.h"
#include "clang/Driver/Option.h"

using namespace clang::driver;

ArgList::ArgList(const char **ArgBegin, const char **ArgEnd) {
  ArgStrings.append(ArgBegin, ArgEnd);
}

ArgList::~ArgList() {
  for (iterator it = begin(), ie = end(); it != ie; ++ie)
    delete *it;
}

void ArgList::append(Arg *A) {
  if (A->getOption().isUnsupported()) {
    assert(0 && "FIXME: unsupported unsupported.");
  }

  Args.push_back(A);
}

Arg *ArgList::getLastArg(options::ID Id) const {
  // FIXME: Make search efficient?

  // FIXME: This needs to not require loading of the option.
  for (const_iterator it = begin(), ie = end(); it != ie; ++it)
    if ((*it)->getOption().matches(Id))
      return *it;
  
  return 0;
}

unsigned ArgList::MakeIndex(const char *String0) {
  unsigned Index = ArgStrings.size();

  // Tuck away so we have a reliable const char *.
  SynthesizedStrings.push_back(String0);
  ArgStrings.push_back(SynthesizedStrings.back().c_str());

  return Index;
}

unsigned ArgList::MakeIndex(const char *String0, const char *String1) {
  unsigned Index0 = MakeIndex(String0);
  unsigned Index1 = MakeIndex(String1);
  assert(Index0 == Index1 && "Unexpected non-consecutive indices!");
  (void) Index1;
  return Index0;
}

Arg *ArgList::MakeFlagArg(const Option *Opt) {
  return new FlagArg(Opt, MakeIndex(Opt->getName()));
}

Arg *ArgList::MakePositionalArg(const Option *Opt, const char *Value) {
  return new PositionalArg(Opt, MakeIndex(Value));
}

Arg *ArgList::MakeSeparateArg(const Option *Opt, const char *Value) {
  return new SeparateArg(Opt, MakeIndex(Opt->getName(), Value), 1);
}

Arg *ArgList::MakeJoinedArg(const Option *Opt, const char *Value) {
  std::string Joined(Opt->getName());
  Joined += Value;
  return new JoinedArg(Opt, MakeIndex(Joined.c_str()));
}
