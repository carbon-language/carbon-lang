//===--- Arg.cpp - Argument Implementations -----------------------------*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "clang/Driver/Arg.h"
#include "clang/Driver/Option.h"
#include "llvm/Support/raw_ostream.h"

using namespace clang::driver;

Arg::Arg(ArgClass _Kind, const Option *_Opt, unsigned _Index) 
  : Kind(_Kind),
    Opt(_Opt),
    Index(_Index)
{
}

Arg::~Arg() { }

void Arg::dump() const {
  llvm::errs() << "<";
  switch (Kind) {
  default:
    assert(0 && "Invalid kind");
#define P(N) case N: llvm::errs() << #N; break
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

  if (const CommaJoinedArg *CJA = dyn_cast<CommaJoinedArg>(this))
    llvm::errs() << " NumValues:" << CJA->getNumValues();

  llvm::errs() << ">\n";

  llvm::errs().flush(); // FIXME
}

PositionalArg::PositionalArg(const Option *Opt, unsigned Index)
  : Arg(PositionalClass, Opt, Index) {
}

void PositionalArg::render(const ArgList &Args, ArgStringList &Output) const {
  assert(0 && "FIXME: Implement");
}

const char *PositionalArg::getValue(const ArgList &Args, unsigned N) const {
  assert(0 && "FIXME: Implement");
}

JoinedArg::JoinedArg(const Option *Opt, unsigned Index)
  : Arg(JoinedClass, Opt, Index) {
}

void JoinedArg::render(const ArgList &Args, ArgStringList &Output) const {
  assert(0 && "FIXME: Implement");
}

const char *JoinedArg::getValue(const ArgList &Args, unsigned N) const {
  assert(0 && "FIXME: Implement");
}

CommaJoinedArg::CommaJoinedArg(const Option *Opt, unsigned Index, 
                               unsigned _NumValues)
  : Arg(CommaJoinedClass, Opt, Index), NumValues(_NumValues) {
}

void CommaJoinedArg::render(const ArgList &Args, ArgStringList &Output) const {
  assert(0 && "FIXME: Implement");
}

const char *CommaJoinedArg::getValue(const ArgList &Args, unsigned N) const {
  assert(0 && "FIXME: Implement");
}

SeparateArg::SeparateArg(const Option *Opt, unsigned Index, unsigned _NumValues)
  : Arg(SeparateClass, Opt, Index), NumValues(_NumValues) {
}

void SeparateArg::render(const ArgList &Args, ArgStringList &Output) const {
  assert(0 && "FIXME: Implement");
}

const char *SeparateArg::getValue(const ArgList &Args, unsigned N) const { 
  assert(0 && "FIXME: Implement");
}

JoinedAndSeparateArg::JoinedAndSeparateArg(const Option *Opt, unsigned Index)
  : Arg(JoinedAndSeparateClass, Opt, Index) {
}

void JoinedAndSeparateArg::render(const ArgList &Args, 
                                  ArgStringList &Output) const {
  assert(0 && "FIXME: Implement");
}

const char *JoinedAndSeparateArg::getValue(const ArgList &Args, 
                                           unsigned N) const {
  assert(0 && "FIXME: Implement");
}
