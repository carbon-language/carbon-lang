//===- IRObjectFile.cpp - IR object file implementation ---------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Part of the IRObjectFile class implementation.
//
//===----------------------------------------------------------------------===//

#include "llvm/Bitcode/ReaderWriter.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Object/IRObjectFile.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;
using namespace object;

IRObjectFile::IRObjectFile(MemoryBuffer *Object, error_code &EC,
                           LLVMContext &Context, bool BufferOwned)
    : SymbolicFile(Binary::ID_IR, Object, BufferOwned) {
  ErrorOr<Module*> MOrErr = parseBitcodeFile(Object, Context);
  if ((EC = MOrErr.getError()))
    return;

  M.reset(MOrErr.get());
}

static const GlobalValue &getGV(DataRefImpl &Symb) {
  return *reinterpret_cast<GlobalValue*>(Symb.p & ~uintptr_t(3));
}

static uintptr_t skipEmpty(Module::const_alias_iterator I, const Module &M) {
  if (I == M.alias_end())
    return 3;
  const GlobalValue *GV = &*I;
  return reinterpret_cast<uintptr_t>(GV) | 2;
}

static uintptr_t skipEmpty(Module::const_global_iterator I, const Module &M) {
  if (I == M.global_end())
    return skipEmpty(M.alias_begin(), M);
  const GlobalValue *GV = &*I;
  return reinterpret_cast<uintptr_t>(GV) | 1;
}

static uintptr_t skipEmpty(Module::const_iterator I, const Module &M) {
  if (I == M.end())
    return skipEmpty(M.global_begin(), M);
  const GlobalValue *GV = &*I;
  return reinterpret_cast<uintptr_t>(GV) | 0;
}

void IRObjectFile::moveSymbolNext(DataRefImpl &Symb) const {
  const GlobalValue *GV = &getGV(Symb);
  const Module &M = *GV->getParent();
  uintptr_t Res;
  switch (Symb.p & 3) {
  case 0: {
    Module::const_iterator Iter(static_cast<const Function*>(GV));
    ++Iter;
    Res = skipEmpty(Iter, M);
    break;
  }
  case 1: {
    Module::const_global_iterator Iter(static_cast<const GlobalVariable*>(GV));
    ++Iter;
    Res = skipEmpty(Iter, M);
    break;
  }
  case 2: {
    Module::const_alias_iterator Iter(static_cast<const GlobalAlias*>(GV));
    ++Iter;
    Res = skipEmpty(Iter, M);
    break;
  }
  case 3:
    llvm_unreachable("Invalid symbol reference");
  }

  Symb.p = Res;
}

error_code IRObjectFile::printSymbolName(raw_ostream &OS,
                                         DataRefImpl Symb) const {
  // FIXME: This should use the Mangler.
  const GlobalValue &GV = getGV(Symb);
  OS << GV.getName();
  return object_error::success;
}

uint32_t IRObjectFile::getSymbolFlags(DataRefImpl Symb) const {
  const GlobalValue &GV = getGV(Symb);

  uint32_t Res = BasicSymbolRef::SF_None;
  if (GV.isDeclaration() || GV.hasAvailableExternallyLinkage())
    Res |= BasicSymbolRef::SF_Undefined;
  if (GV.hasPrivateLinkage() || GV.hasLinkerPrivateLinkage() ||
      GV.hasLinkerPrivateWeakLinkage())
    Res |= BasicSymbolRef::SF_FormatSpecific;
  if (!GV.hasLocalLinkage())
    Res |= BasicSymbolRef::SF_Global;
  if (GV.hasCommonLinkage())
    Res |= BasicSymbolRef::SF_Common;
  if (GV.hasLinkOnceLinkage() || GV.hasWeakLinkage())
    Res |= BasicSymbolRef::SF_Weak;

  return Res;
}

const GlobalValue &IRObjectFile::getSymbolGV(DataRefImpl Symb) const {
  const GlobalValue &GV = getGV(Symb);
  return GV;
}

basic_symbol_iterator IRObjectFile::symbol_begin_impl() const {
  Module::const_iterator I = M->begin();
  DataRefImpl Ret;
  Ret.p = skipEmpty(I, *M);
  return basic_symbol_iterator(BasicSymbolRef(Ret, this));
}

basic_symbol_iterator IRObjectFile::symbol_end_impl() const {
  DataRefImpl Ret;
  Ret.p = 3;
  return basic_symbol_iterator(BasicSymbolRef(Ret, this));
}

ErrorOr<SymbolicFile *> llvm::object::SymbolicFile::createIRObjectFile(
    MemoryBuffer *Object, LLVMContext &Context, bool BufferOwned) {
  error_code EC;
  OwningPtr<IRObjectFile> Ret(
      new IRObjectFile(Object, EC, Context, BufferOwned));
  if (EC)
    return EC;
  return Ret.take();
}
