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
#include "llvm/Support/raw_ostream.h"

using namespace clang::driver;

Arg::Arg(ArgClass _Kind, const Option *_Opt, unsigned _Index, 
         const Arg *_BaseArg) 
  : Kind(_Kind), Opt(_Opt), BaseArg(_BaseArg), Index(_Index), Claimed(false)
{
}

Arg::~Arg() { }

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
  Output.push_back(Args.getArgString(getIndex()));
}

const char *FlagArg::getValue(const ArgList &Args, unsigned N) const {
  assert(0 && "Invalid index.");
  return 0;
}

PositionalArg::PositionalArg(const Option *Opt, unsigned Index, 
                             const Arg *BaseArg)
  : Arg(PositionalClass, Opt, Index, BaseArg) {
}

void PositionalArg::render(const ArgList &Args, ArgStringList &Output) const {
  Output.push_back(Args.getArgString(getIndex()));
}

const char *PositionalArg::getValue(const ArgList &Args, unsigned N) const {
  assert(N < getNumValues() && "Invalid index.");
  return Args.getArgString(getIndex());
}

JoinedArg::JoinedArg(const Option *Opt, unsigned Index, const Arg *BaseArg)
  : Arg(JoinedClass, Opt, Index, BaseArg) {
}

void JoinedArg::render(const ArgList &Args, ArgStringList &Output) const {
  if (getOption().hasForceSeparateRender()) {
    Output.push_back(getOption().getName());
    Output.push_back(getValue(Args, 0));
  } else {
    Output.push_back(Args.getArgString(getIndex()));
  }
}

const char *JoinedArg::getValue(const ArgList &Args, unsigned N) const {
  assert(N < getNumValues() && "Invalid index.");
  // FIXME: Avoid strlen.
  return Args.getArgString(getIndex()) + strlen(getOption().getName());
}

CommaJoinedArg::CommaJoinedArg(const Option *Opt, unsigned Index, 
                               const char *Str, const Arg *BaseArg)
  : Arg(CommaJoinedClass, Opt, Index, BaseArg) {
  const char *Prev = Str;  
  for (;; ++Str) {
    char c = *Str;

    if (!c) {
      if (Prev != Str)
        Values.push_back(std::string(Prev, Str));
      break;
    } else if (c == ',') {
      if (Prev != Str)
        Values.push_back(std::string(Prev, Str));
      Prev = Str + 1;
    }
  }
}

void CommaJoinedArg::render(const ArgList &Args, ArgStringList &Output) const {
  Output.push_back(Args.getArgString(getIndex()));
}

const char *CommaJoinedArg::getValue(const ArgList &Args, unsigned N) const {
  assert(N < getNumValues() && "Invalid index.");
  return Values[N].c_str();
}

SeparateArg::SeparateArg(const Option *Opt, unsigned Index, unsigned _NumValues,
                         const Arg *BaseArg)
  : Arg(SeparateClass, Opt, Index, BaseArg), NumValues(_NumValues) {
}

void SeparateArg::render(const ArgList &Args, ArgStringList &Output) const {
  if (getOption().hasForceJoinedRender()) {
    assert(getNumValues() == 1 && "Cannot force joined render with > 1 args.");
    // FIXME: Avoid std::string.
    std::string Joined(getOption().getName());
    Joined += Args.getArgString(getIndex());
    Output.push_back(Args.MakeArgString(Joined.c_str()));
  } else {
    Output.push_back(Args.getArgString(getIndex()));
    for (unsigned i = 0; i < NumValues; ++i)
      Output.push_back(Args.getArgString(getIndex() + 1 + i));
  }
}

const char *SeparateArg::getValue(const ArgList &Args, unsigned N) const { 
  assert(N < getNumValues() && "Invalid index.");
  return Args.getArgString(getIndex() + 1 + N);
}

JoinedAndSeparateArg::JoinedAndSeparateArg(const Option *Opt, unsigned Index, 
                                           const Arg *BaseArg)
  : Arg(JoinedAndSeparateClass, Opt, Index, BaseArg) {
}

void JoinedAndSeparateArg::render(const ArgList &Args, 
                                  ArgStringList &Output) const {
  Output.push_back(Args.getArgString(getIndex()));
  Output.push_back(Args.getArgString(getIndex() + 1));
}

const char *JoinedAndSeparateArg::getValue(const ArgList &Args, 
                                           unsigned N) const {
  assert(N < getNumValues() && "Invalid index.");
  if (N == 0)
    return Args.getArgString(getIndex()) + strlen(getOption().getName());
  return Args.getArgString(getIndex() + 1);
}
