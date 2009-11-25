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

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/raw_ostream.h"

using namespace clang::driver;

void arg_iterator::SkipToNextArg() {
  for (; Current != Args.end(); ++Current) {
    // Done if there are no filters.
    if (!Id0.isValid())
      break;

    // Otherwise require a match.
    const Option &O = (*Current)->getOption();
    if (O.matches(Id0) ||
        (Id1.isValid() && O.matches(Id1)) ||
        (Id2.isValid() && O.matches(Id2)))
      break;
  }
}

//

ArgList::ArgList(arglist_type &_Args) : Args(_Args) {
}

ArgList::~ArgList() {
}

void ArgList::append(Arg *A) {
  Args.push_back(A);
}

Arg *ArgList::getLastArgNoClaim(OptSpecifier Id) const {
  // FIXME: Make search efficient?
  for (const_reverse_iterator it = rbegin(), ie = rend(); it != ie; ++it)
    if ((*it)->getOption().matches(Id))
      return *it;
  return 0;
}

Arg *ArgList::getLastArg(OptSpecifier Id) const {
  Arg *A = getLastArgNoClaim(Id);
  if (A)
    A->claim();
  return A;
}

Arg *ArgList::getLastArg(OptSpecifier Id0, OptSpecifier Id1) const {
  Arg *Res, *A0 = getLastArgNoClaim(Id0), *A1 = getLastArgNoClaim(Id1);

  if (A0 && A1)
    Res = A0->getIndex() > A1->getIndex() ? A0 : A1;
  else
    Res = A0 ? A0 : A1;

  if (Res)
    Res->claim();

  return Res;
}

Arg *ArgList::getLastArg(OptSpecifier Id0, OptSpecifier Id1,
                         OptSpecifier Id2) const {
  Arg *Res = 0;
  Arg *A0 = getLastArgNoClaim(Id0);
  Arg *A1 = getLastArgNoClaim(Id1);
  Arg *A2 = getLastArgNoClaim(Id2);

  int A0Idx = A0 ? A0->getIndex() : -1;
  int A1Idx = A1 ? A1->getIndex() : -1;
  int A2Idx = A2 ? A2->getIndex() : -1;

  if (A0Idx > A1Idx) {
    if (A0Idx > A2Idx)
      Res = A0;
    else if (A2Idx != -1)
      Res = A2;
  } else {
    if (A1Idx > A2Idx)
      Res = A1;
    else if (A2Idx != -1)
      Res = A2;
  }

  if (Res)
    Res->claim();

  return Res;
}

bool ArgList::hasFlag(OptSpecifier Pos, OptSpecifier Neg, bool Default) const {
  if (Arg *A = getLastArg(Pos, Neg))
    return A->getOption().matches(Pos);
  return Default;
}

void ArgList::AddLastArg(ArgStringList &Output, OptSpecifier Id) const {
  if (Arg *A = getLastArg(Id)) {
    A->claim();
    A->render(*this, Output);
  }
}

void ArgList::AddAllArgs(ArgStringList &Output, OptSpecifier Id0,
                         OptSpecifier Id1, OptSpecifier Id2) const {
  for (arg_iterator it = filtered_begin(Id0, Id1, Id2),
         ie = filtered_end(); it != ie; ++it) {
    it->claim();
    it->render(*this, Output);
  }
}

void ArgList::AddAllArgValues(ArgStringList &Output, OptSpecifier Id0,
                              OptSpecifier Id1, OptSpecifier Id2) const {
  for (arg_iterator it = filtered_begin(Id0, Id1, Id2),
         ie = filtered_end(); it != ie; ++it) {
    it->claim();
    for (unsigned i = 0, e = it->getNumValues(); i != e; ++i)
      Output.push_back(it->getValue(*this, i));
  }
}

void ArgList::AddAllArgsTranslated(ArgStringList &Output, OptSpecifier Id0,
                                   const char *Translation,
                                   bool Joined) const {
  for (arg_iterator it = filtered_begin(Id0),
         ie = filtered_end(); it != ie; ++it) {
    it->claim();

    if (Joined) {
      Output.push_back(MakeArgString(llvm::StringRef(Translation) +
                                     it->getValue(*this, 0)));
    } else {
      Output.push_back(Translation);
      Output.push_back(it->getValue(*this, 0));
    }
  }
}

void ArgList::ClaimAllArgs(OptSpecifier Id0) const {
  for (arg_iterator it = filtered_begin(Id0),
         ie = filtered_end(); it != ie; ++it)
      it->claim();
}

const char *ArgList::MakeArgString(const llvm::Twine &T) const {
  llvm::SmallString<256> Str;
  T.toVector(Str);
  return MakeArgString(Str.str());
}

//

InputArgList::InputArgList(const char **ArgBegin, const char **ArgEnd)
  : ArgList(ActualArgs), NumInputArgStrings(ArgEnd - ArgBegin) {
  ArgStrings.append(ArgBegin, ArgEnd);
}

InputArgList::~InputArgList() {
  // An InputArgList always owns its arguments.
  for (iterator it = begin(), ie = end(); it != ie; ++it)
    delete *it;
}

unsigned InputArgList::MakeIndex(llvm::StringRef String0) const {
  unsigned Index = ArgStrings.size();

  // Tuck away so we have a reliable const char *.
  SynthesizedStrings.push_back(String0);
  ArgStrings.push_back(SynthesizedStrings.back().c_str());

  return Index;
}

unsigned InputArgList::MakeIndex(llvm::StringRef String0,
                                 llvm::StringRef String1) const {
  unsigned Index0 = MakeIndex(String0);
  unsigned Index1 = MakeIndex(String1);
  assert(Index0 + 1 == Index1 && "Unexpected non-consecutive indices!");
  (void) Index1;
  return Index0;
}

const char *InputArgList::MakeArgString(llvm::StringRef Str) const {
  return getArgString(MakeIndex(Str));
}

//

DerivedArgList::DerivedArgList(InputArgList &_BaseArgs, bool _OnlyProxy)
  : ArgList(_OnlyProxy ? _BaseArgs.getArgs() : ActualArgs),
    BaseArgs(_BaseArgs), OnlyProxy(_OnlyProxy) {
}

DerivedArgList::~DerivedArgList() {
  // We only own the arguments we explicitly synthesized.
  for (iterator it = SynthesizedArgs.begin(), ie = SynthesizedArgs.end();
       it != ie; ++it)
    delete *it;
}

const char *DerivedArgList::MakeArgString(llvm::StringRef Str) const {
  return BaseArgs.MakeArgString(Str);
}

Arg *DerivedArgList::MakeFlagArg(const Arg *BaseArg, const Option *Opt) const {
  return new FlagArg(Opt, BaseArgs.MakeIndex(Opt->getName()), BaseArg);
}

Arg *DerivedArgList::MakePositionalArg(const Arg *BaseArg, const Option *Opt,
                                       llvm::StringRef Value) const {
  return new PositionalArg(Opt, BaseArgs.MakeIndex(Value), BaseArg);
}

Arg *DerivedArgList::MakeSeparateArg(const Arg *BaseArg, const Option *Opt,
                                     llvm::StringRef Value) const {
  return new SeparateArg(Opt, BaseArgs.MakeIndex(Opt->getName(), Value), 1,
                         BaseArg);
}

Arg *DerivedArgList::MakeJoinedArg(const Arg *BaseArg, const Option *Opt,
                                   llvm::StringRef Value) const {
  std::string Joined(Opt->getName());
  Joined += Value;
  return new JoinedArg(Opt, BaseArgs.MakeIndex(Joined.c_str()), BaseArg);
}
