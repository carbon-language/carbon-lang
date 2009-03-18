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

Arg::Arg(ArgClass _Kind, const Option *_Opt, unsigned _Index) 
  : Kind(_Kind),
    Opt(_Opt),
    Index(_Index),
    Claimed(false)
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

FlagArg::FlagArg(const Option *Opt, unsigned Index)
  : Arg(FlagClass, Opt, Index) {
}

void FlagArg::render(const ArgList &Args, ArgStringList &Output) const {
  Output.push_back(Args.getArgString(getIndex()));
}

const char *FlagArg::getValue(const ArgList &Args, unsigned N) const {
  assert(0 && "Invalid index.");
  return 0;
}

PositionalArg::PositionalArg(const Option *Opt, unsigned Index)
  : Arg(PositionalClass, Opt, Index) {
}

void PositionalArg::render(const ArgList &Args, ArgStringList &Output) const {
  Output.push_back(Args.getArgString(getIndex()));
}

const char *PositionalArg::getValue(const ArgList &Args, unsigned N) const {
  assert(N < getNumValues() && "Invalid index.");
  return Args.getArgString(getIndex());
}

JoinedArg::JoinedArg(const Option *Opt, unsigned Index)
  : Arg(JoinedClass, Opt, Index) {
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
                               const char *Str)
  : Arg(CommaJoinedClass, Opt, Index) {
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

SeparateArg::SeparateArg(const Option *Opt, unsigned Index, unsigned _NumValues)
  : Arg(SeparateClass, Opt, Index), NumValues(_NumValues) {
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

JoinedAndSeparateArg::JoinedAndSeparateArg(const Option *Opt, unsigned Index)
  : Arg(JoinedAndSeparateClass, Opt, Index) {
}

void JoinedAndSeparateArg::render(const ArgList &Args, 
                                  ArgStringList &Output) const {
  Output.push_back(Args.getArgString(getIndex()));
  Output.push_back(Args.getArgString(getIndex()) + 1);
}

const char *JoinedAndSeparateArg::getValue(const ArgList &Args, 
                                           unsigned N) const {
  assert(N < getNumValues() && "Invalid index.");
  if (N == 0)
    return Args.getArgString(getIndex()) + strlen(getOption().getName());
  return Args.getArgString(getIndex() + 1);
}
