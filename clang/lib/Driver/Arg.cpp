//===--- Arg.cpp - Argument Implementations -----------------------------*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "clang/Driver/Arg.h"
#include "clang/Driver/ArgList.h"
#include "clang/Driver/Option.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/raw_ostream.h"

using namespace clang::driver;

Arg::Arg(ArgClass _Kind, const Option *_Opt, unsigned _Index,
         const Arg *_BaseArg)
  : Kind(_Kind), Opt(_Opt), BaseArg(_BaseArg), Index(_Index),
    Claimed(false), OwnsValues(false) {
}

Arg::~Arg() {
  if (OwnsValues) {
    for (unsigned i = 0, e = Values.size(); i != e; ++i)
      delete[] Values[i];
  }
}

void Arg::dump() const {
  llvm::errs() << "<";
  switch (Kind) {
  default:
    assert(0 && "Invalid kind");
#define P(N) case N: llvm::errs() << #N; break
    P(FlagClass);
    P(PositionalClass);
    P(JoinedClass);
    P(SeparateClass);
    P(CommaJoinedClass);
    P(JoinedAndSeparateClass);
#undef P
  }

  llvm::errs() << " Opt:";
  Opt->dump();

  llvm::errs() << " Index:" << Index;

  if (isa<CommaJoinedArg>(this) || isa<SeparateArg>(this))
    llvm::errs() << " NumValues:" << getNumValues();

  llvm::errs() << ">\n";
}

std::string Arg::getAsString(const ArgList &Args) const {
  std::string Res;
  llvm::raw_string_ostream OS(Res);

  ArgStringList ASL;
  render(Args, ASL);
  for (ArgStringList::iterator
         it = ASL.begin(), ie = ASL.end(); it != ie; ++it) {
    if (it != ASL.begin())
      OS << ' ';
    OS << *it;
  }

  return OS.str();
}

void Arg::renderAsInput(const ArgList &Args, ArgStringList &Output) const {
  if (!getOption().hasNoOptAsInput()) {
    render(Args, Output);
    return;
  }

  for (unsigned i = 0, e = getNumValues(); i != e; ++i)
    Output.push_back(getValue(Args, i));
}

FlagArg::FlagArg(const Option *Opt, unsigned Index, const Arg *BaseArg)
  : Arg(FlagClass, Opt, Index, BaseArg) {
}

void FlagArg::render(const ArgList &Args, ArgStringList &Output) const {
  Output.push_back(getOption().getName());
}

PositionalArg::PositionalArg(const Option *Opt, unsigned Index,
                             const char *Value0, const Arg *BaseArg)
  : Arg(PositionalClass, Opt, Index, BaseArg) {
  getValues().push_back(Value0);
}

void PositionalArg::render(const ArgList &Args, ArgStringList &Output) const {
  Output.push_back(Args.getArgString(getIndex()));
}

JoinedArg::JoinedArg(const Option *Opt, unsigned Index, const char *Value0,
                     const Arg *BaseArg)
  : Arg(JoinedClass, Opt, Index, BaseArg) {
  getValues().push_back(Value0);
}

void JoinedArg::render(const ArgList &Args, ArgStringList &Output) const {
  if (getOption().hasForceSeparateRender()) {
    Output.push_back(getOption().getName());
    Output.push_back(getValue(Args, 0));
  } else {
    Output.push_back(Args.GetOrMakeJoinedArgString(
                       getIndex(), getOption().getName(), getValue(Args, 0)));
  }
}

CommaJoinedArg::CommaJoinedArg(const Option *Opt, unsigned Index,
                               const char *Str, const Arg *BaseArg)
  : Arg(CommaJoinedClass, Opt, Index, BaseArg) {
  const char *Prev = Str;
  for (;; ++Str) {
    char c = *Str;

    if (!c || c == ',') {
      if (Prev != Str) {
        char *Value = new char[Str - Prev + 1];
        memcpy(Value, Prev, Str - Prev);
        Value[Str - Prev] = '\0';
        getValues().push_back(Value);
      }

      if (!c)
        break;

      Prev = Str + 1;
    }
  }

  setOwnsValues(true);
}

void CommaJoinedArg::render(const ArgList &Args, ArgStringList &Output) const {
  Output.push_back(Args.getArgString(getIndex()));
}

SeparateArg::SeparateArg(const Option *Opt, unsigned Index, const char *Value0,
                         const Arg *BaseArg)
  : Arg(SeparateClass, Opt, Index, BaseArg) {
  getValues().push_back(Value0);
}

void SeparateArg::render(const ArgList &Args, ArgStringList &Output) const {
  if (getOption().hasForceJoinedRender()) {
    assert(getNumValues() == 1 && "Cannot force joined render with > 1 args.");
    Output.push_back(Args.MakeArgString(llvm::StringRef(getOption().getName()) +
                                        getValue(Args, 0)));
  } else {
    Output.push_back(getOption().getName());
    for (unsigned i = 0; i != getNumValues(); ++i)
      Output.push_back(getValue(Args, i));
  }
}

JoinedAndSeparateArg::JoinedAndSeparateArg(const Option *Opt, unsigned Index,
                                           const char *Value0,
                                           const char *Value1,
                                           const Arg *BaseArg)
  : Arg(JoinedAndSeparateClass, Opt, Index, BaseArg) {
  getValues().push_back(Value0);
  getValues().push_back(Value1);
}

void JoinedAndSeparateArg::render(const ArgList &Args,
                                  ArgStringList &Output) const {
  Output.push_back(Args.GetOrMakeJoinedArgString(
                     getIndex(), getOption().getName(), getValue(Args, 0)));
  Output.push_back(getValue(Args, 1));
}
