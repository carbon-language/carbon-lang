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

ArgList::ArgList(arglist_type &_Args) : Args(_Args) {
}

ArgList::~ArgList() {
}

void ArgList::append(Arg *A) {
  Args.push_back(A);
}

Arg *ArgList::getLastArg(options::ID Id, bool Claim) const {
  // FIXME: Make search efficient?
  for (const_reverse_iterator it = rbegin(), ie = rend(); it != ie; ++it) {
    if ((*it)->getOption().matches(Id)) {
      if (Claim) (*it)->claim();
      return *it;
    }
  }
  
  return 0;
}

Arg *ArgList::getLastArg(options::ID Id0, options::ID Id1, bool Claim) const {
  Arg *Res, *A0 = getLastArg(Id0, false), *A1 = getLastArg(Id1, false);
  
  if (A0 && A1)
    Res = A0->getIndex() > A1->getIndex() ? A0 : A1;
  else
    Res = A0 ? A0 : A1;

  if (Claim && Res)
    Res->claim();

  return Res;
}

bool ArgList::hasFlag(options::ID Pos, options::ID Neg, bool Default) const {
  if (Arg *A = getLastArg(Pos, Neg))
    return A->getOption().matches(Pos);
  return Default;
}

void ArgList::AddLastArg(ArgStringList &Output, options::ID Id) const {
  if (Arg *A = getLastArg(Id)) {
    A->claim();
    A->render(*this, Output);
  }
}

void ArgList::AddAllArgs(ArgStringList &Output, options::ID Id0) const {
  // FIXME: Make fast.
  for (const_iterator it = begin(), ie = end(); it != ie; ++it) {
    const Arg *A = *it;
    if (A->getOption().matches(Id0)) {
      A->claim();
      A->render(*this, Output);
    }
  }
}

void ArgList::AddAllArgs(ArgStringList &Output, options::ID Id0, 
                         options::ID Id1) const {
  // FIXME: Make fast.
  for (const_iterator it = begin(), ie = end(); it != ie; ++it) {
    const Arg *A = *it;
    if (A->getOption().matches(Id0) || A->getOption().matches(Id1)) {
      A->claim();
      A->render(*this, Output);
    }
  }
}

void ArgList::AddAllArgs(ArgStringList &Output, options::ID Id0, 
                         options::ID Id1, options::ID Id2) const {
  // FIXME: Make fast.
  for (const_iterator it = begin(), ie = end(); it != ie; ++it) {
    const Arg *A = *it;
    if (A->getOption().matches(Id0) || A->getOption().matches(Id1) ||
        A->getOption().matches(Id2)) {
      A->claim();
      A->render(*this, Output);
    }
  }
}

void ArgList::AddAllArgValues(ArgStringList &Output, options::ID Id0) const {
  // FIXME: Make fast.
  for (const_iterator it = begin(), ie = end(); it != ie; ++it) {
    const Arg *A = *it;
    if (A->getOption().matches(Id0)) {
      A->claim();
      for (unsigned i = 0, e = A->getNumValues(); i != e; ++i)
        Output.push_back(A->getValue(*this, i));
    }
  }
}

void ArgList::AddAllArgValues(ArgStringList &Output, options::ID Id0, 
                              options::ID Id1) const {
  // FIXME: Make fast.
  for (const_iterator it = begin(), ie = end(); it != ie; ++it) {
    const Arg *A = *it;
    if (A->getOption().matches(Id0) || A->getOption().matches(Id1)) {
      A->claim();
      for (unsigned i = 0, e = A->getNumValues(); i != e; ++i)
        Output.push_back(A->getValue(*this, i));
    }
  }
}

void ArgList::AddAllArgsTranslated(ArgStringList &Output, options::ID Id0,
                                   const char *Translation,
                                   bool Joined) const {
  // FIXME: Make fast.
  for (const_iterator it = begin(), ie = end(); it != ie; ++it) {
    const Arg *A = *it;
    if (A->getOption().matches(Id0)) {
      A->claim();

      if (Joined) {
        std::string Value = Translation;
        Value += A->getValue(*this, 0);
        Output.push_back(MakeArgString(Value.c_str()));
      } else {
        Output.push_back(Translation);
        Output.push_back(A->getValue(*this, 0));
      }
    }
  }
}

void ArgList::ClaimAllArgs(options::ID Id0) const {
  // FIXME: Make fast.
  for (const_iterator it = begin(), ie = end(); it != ie; ++it) {
    const Arg *A = *it;
    if (A->getOption().matches(Id0))
      A->claim();
  }
}

//

InputArgList::InputArgList(const char **ArgBegin, const char **ArgEnd) 
  : ArgList(ActualArgs), NumInputArgStrings(ArgEnd - ArgBegin) 
{
  ArgStrings.append(ArgBegin, ArgEnd);
}

InputArgList::~InputArgList() {
  // An InputArgList always owns its arguments.
  for (iterator it = begin(), ie = end(); it != ie; ++it)
    delete *it;
}

unsigned InputArgList::MakeIndex(const char *String0) const {
  unsigned Index = ArgStrings.size();

  // Tuck away so we have a reliable const char *.
  SynthesizedStrings.push_back(String0);
  ArgStrings.push_back(SynthesizedStrings.back().c_str());

  return Index;
}

unsigned InputArgList::MakeIndex(const char *String0, 
                                 const char *String1) const {
  unsigned Index0 = MakeIndex(String0);
  unsigned Index1 = MakeIndex(String1);
  assert(Index0 + 1 == Index1 && "Unexpected non-consecutive indices!");
  (void) Index1;
  return Index0;
}

const char *InputArgList::MakeArgString(const char *Str) const {
  return getArgString(MakeIndex(Str));
}

//

DerivedArgList::DerivedArgList(InputArgList &_BaseArgs, bool _OnlyProxy)
  : ArgList(_OnlyProxy ? _BaseArgs.getArgs() : ActualArgs),
    BaseArgs(_BaseArgs), OnlyProxy(_OnlyProxy)
{
}

DerivedArgList::~DerivedArgList() {
  // We only own the arguments we explicitly synthesized.
  for (iterator it = SynthesizedArgs.begin(), ie = SynthesizedArgs.end(); 
       it != ie; ++it)
    delete *it;
}

const char *DerivedArgList::MakeArgString(const char *Str) const {
  return BaseArgs.MakeArgString(Str);
}

Arg *DerivedArgList::MakeFlagArg(const Arg *BaseArg, const Option *Opt) const {
  return new FlagArg(Opt, BaseArgs.MakeIndex(Opt->getName()), BaseArg);
}

Arg *DerivedArgList::MakePositionalArg(const Arg *BaseArg, const Option *Opt, 
                                       const char *Value) const {
  return new PositionalArg(Opt, BaseArgs.MakeIndex(Value), BaseArg);
}

Arg *DerivedArgList::MakeSeparateArg(const Arg *BaseArg, const Option *Opt, 
                                     const char *Value) const {
  return new SeparateArg(Opt, BaseArgs.MakeIndex(Opt->getName(), Value), 1, 
                         BaseArg);
}

Arg *DerivedArgList::MakeJoinedArg(const Arg *BaseArg, const Option *Opt, 
                                   const char *Value) const {
  std::string Joined(Opt->getName());
  Joined += Value;
  return new JoinedArg(Opt, BaseArgs.MakeIndex(Joined.c_str()), BaseArg);
}
