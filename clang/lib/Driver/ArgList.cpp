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
#include "clang/Driver/DriverDiagnostic.h"
#include "clang/Driver/Option.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/raw_ostream.h"

using namespace clang;
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

ArgList::ArgList() {
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
  Arg *Res = 0;
  for (const_reverse_iterator it = rbegin(), ie = rend(); it != ie; ++it) {
    if ((*it)->getOption().matches(Id0) ||
        (*it)->getOption().matches(Id1)) {
      Res = *it;
      break;
    }
  }

  if (Res)
    Res->claim();

  return Res;
}

Arg *ArgList::getLastArg(OptSpecifier Id0, OptSpecifier Id1,
                         OptSpecifier Id2) const {
  Arg *Res = 0;
  for (const_reverse_iterator it = rbegin(), ie = rend(); it != ie; ++it) {
    if ((*it)->getOption().matches(Id0) ||
        (*it)->getOption().matches(Id1) ||
        (*it)->getOption().matches(Id2)) {
      Res = *it;
      break;
    }
  }

  if (Res)
    Res->claim();

  return Res;
}

Arg *ArgList::getLastArg(OptSpecifier Id0, OptSpecifier Id1,
                         OptSpecifier Id2, OptSpecifier Id3) const {
  Arg *Res = 0;
  for (const_reverse_iterator it = rbegin(), ie = rend(); it != ie; ++it) {
    if ((*it)->getOption().matches(Id0) ||
        (*it)->getOption().matches(Id1) ||
        (*it)->getOption().matches(Id2) ||
        (*it)->getOption().matches(Id3)) {
      Res = *it;
      break;
    }
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

llvm::StringRef ArgList::getLastArgValue(OptSpecifier Id,
                                         llvm::StringRef Default) const {
  if (Arg *A = getLastArg(Id))
    return A->getValue(*this);
  return Default;
}

int ArgList::getLastArgIntValue(OptSpecifier Id, int Default,
                                clang::Diagnostic &Diags) const {
  int Res = Default;

  if (Arg *A = getLastArg(Id)) {
    if (llvm::StringRef(A->getValue(*this)).getAsInteger(10, Res))
      Diags.Report(diag::err_drv_invalid_int_value)
        << A->getAsString(*this) << A->getValue(*this);
  }

  return Res;
}

std::vector<std::string> ArgList::getAllArgValues(OptSpecifier Id) const {
  llvm::SmallVector<const char *, 16> Values;
  AddAllArgValues(Values, Id);
  return std::vector<std::string>(Values.begin(), Values.end());
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
    (*it)->claim();
    (*it)->render(*this, Output);
  }
}

void ArgList::AddAllArgValues(ArgStringList &Output, OptSpecifier Id0,
                              OptSpecifier Id1, OptSpecifier Id2) const {
  for (arg_iterator it = filtered_begin(Id0, Id1, Id2),
         ie = filtered_end(); it != ie; ++it) {
    (*it)->claim();
    for (unsigned i = 0, e = (*it)->getNumValues(); i != e; ++i)
      Output.push_back((*it)->getValue(*this, i));
  }
}

void ArgList::AddAllArgsTranslated(ArgStringList &Output, OptSpecifier Id0,
                                   const char *Translation,
                                   bool Joined) const {
  for (arg_iterator it = filtered_begin(Id0),
         ie = filtered_end(); it != ie; ++it) {
    (*it)->claim();

    if (Joined) {
      Output.push_back(MakeArgString(llvm::StringRef(Translation) +
                                     (*it)->getValue(*this, 0)));
    } else {
      Output.push_back(Translation);
      Output.push_back((*it)->getValue(*this, 0));
    }
  }
}

void ArgList::ClaimAllArgs(OptSpecifier Id0) const {
  for (arg_iterator it = filtered_begin(Id0),
         ie = filtered_end(); it != ie; ++it)
    (*it)->claim();
}

const char *ArgList::MakeArgString(const llvm::Twine &T) const {
  llvm::SmallString<256> Str;
  T.toVector(Str);
  return MakeArgString(Str.str());
}

const char *ArgList::GetOrMakeJoinedArgString(unsigned Index,
                                              llvm::StringRef LHS,
                                              llvm::StringRef RHS) const {
  llvm::StringRef Cur = getArgString(Index);
  if (Cur.size() == LHS.size() + RHS.size() &&
      Cur.startswith(LHS) && Cur.endswith(RHS))
    return Cur.data();

  return MakeArgString(LHS + RHS);
}

//

InputArgList::InputArgList(const char* const *ArgBegin,
                           const char* const *ArgEnd)
  : NumInputArgStrings(ArgEnd - ArgBegin) {
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

DerivedArgList::DerivedArgList(const InputArgList &_BaseArgs)
  : BaseArgs(_BaseArgs) {
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
  Arg *A = new Arg(Opt, BaseArgs.MakeIndex(Opt->getName()), BaseArg);
  SynthesizedArgs.push_back(A);
  return A;
}

Arg *DerivedArgList::MakePositionalArg(const Arg *BaseArg, const Option *Opt,
                                       llvm::StringRef Value) const {
  unsigned Index = BaseArgs.MakeIndex(Value);
  Arg *A = new Arg(Opt, Index, BaseArgs.getArgString(Index), BaseArg);
  SynthesizedArgs.push_back(A);
  return A;
}

Arg *DerivedArgList::MakeSeparateArg(const Arg *BaseArg, const Option *Opt,
                                     llvm::StringRef Value) const {
  unsigned Index = BaseArgs.MakeIndex(Opt->getName(), Value);
  Arg *A = new Arg(Opt, Index, BaseArgs.getArgString(Index + 1), BaseArg);
  SynthesizedArgs.push_back(A);
  return A;
}

Arg *DerivedArgList::MakeJoinedArg(const Arg *BaseArg, const Option *Opt,
                                   llvm::StringRef Value) const {
  unsigned Index = BaseArgs.MakeIndex(Opt->getName() + Value.str());
  Arg *A = new Arg(Opt, Index,
                   BaseArgs.getArgString(Index) + strlen(Opt->getName()),
                   BaseArg);
  SynthesizedArgs.push_back(A);
  return A;
}
