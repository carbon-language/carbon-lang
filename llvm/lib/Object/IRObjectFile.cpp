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

#include "llvm/Object/IRObjectFile.h"
#include "RecordStreamer.h"
#include "llvm/Bitcode/ReaderWriter.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/GVMaterializer.h"
#include "llvm/IR/Mangler.h"
#include "llvm/IR/Module.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCObjectFileInfo.h"
#include "llvm/MC/MCTargetAsmParser.h"
#include "llvm/MC/MCParser/MCAsmParser.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;
using namespace object;

IRObjectFile::IRObjectFile(std::unique_ptr<MemoryBuffer> Object,
                           std::unique_ptr<Module> Mod)
    : SymbolicFile(Binary::ID_IR, std::move(Object)), M(std::move(Mod)) {
  // If we have a DataLayout, setup a mangler.
  const DataLayout *DL = M->getDataLayout();
  if (!DL)
    return;

  Mang.reset(new Mangler(DL));

  const std::string &InlineAsm = M->getModuleInlineAsm();
  if (InlineAsm.empty())
    return;

  StringRef Triple = M->getTargetTriple();
  std::string Err;
  const Target *T = TargetRegistry::lookupTarget(Triple, Err);
  if (!T)
    return;

  std::unique_ptr<MCRegisterInfo> MRI(T->createMCRegInfo(Triple));
  if (!MRI)
    return;

  std::unique_ptr<MCAsmInfo> MAI(T->createMCAsmInfo(*MRI, Triple));
  if (!MAI)
    return;

  std::unique_ptr<MCSubtargetInfo> STI(
      T->createMCSubtargetInfo(Triple, "", ""));
  if (!STI)
    return;

  std::unique_ptr<MCInstrInfo> MCII(T->createMCInstrInfo());
  if (!MCII)
    return;

  MCObjectFileInfo MOFI;
  MCContext MCCtx(MAI.get(), MRI.get(), &MOFI);
  MOFI.InitMCObjectFileInfo(Triple, Reloc::Default, CodeModel::Default, MCCtx);
  std::unique_ptr<RecordStreamer> Streamer(new RecordStreamer(MCCtx));

  std::unique_ptr<MemoryBuffer> Buffer(MemoryBuffer::getMemBuffer(InlineAsm));
  SourceMgr SrcMgr;
  SrcMgr.AddNewSourceBuffer(Buffer.release(), SMLoc());
  std::unique_ptr<MCAsmParser> Parser(
      createMCAsmParser(SrcMgr, MCCtx, *Streamer, *MAI));

  MCTargetOptions MCOptions;
  std::unique_ptr<MCTargetAsmParser> TAP(
      T->createMCAsmParser(*STI, *Parser, *MCII, MCOptions));
  if (!TAP)
    return;

  Parser->setTargetParser(*TAP);
  if (Parser->Run(false))
    return;

  for (auto &KV : *Streamer) {
    StringRef Key = KV.first();
    RecordStreamer::State Value = KV.second;
    uint32_t Res = BasicSymbolRef::SF_None;
    switch (Value) {
    case RecordStreamer::NeverSeen:
      llvm_unreachable("foo");
    case RecordStreamer::DefinedGlobal:
      Res |= BasicSymbolRef::SF_Global;
      break;
    case RecordStreamer::Defined:
      break;
    case RecordStreamer::Global:
    case RecordStreamer::Used:
      Res |= BasicSymbolRef::SF_Undefined;
      Res |= BasicSymbolRef::SF_Global;
      break;
    }
    AsmSymbols.push_back(
        std::make_pair<std::string, uint32_t>(Key, std::move(Res)));
  }
}

IRObjectFile::~IRObjectFile() {
  GVMaterializer *GVM =  M->getMaterializer();
  if (GVM)
    GVM->releaseBuffer();
 }

static const GlobalValue *getGV(DataRefImpl &Symb) {
  if ((Symb.p & 3) == 3)
    return nullptr;

  return reinterpret_cast<GlobalValue*>(Symb.p & ~uintptr_t(3));
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

static unsigned getAsmSymIndex(DataRefImpl Symb) {
  assert((Symb.p & uintptr_t(3)) == 3);
  uintptr_t Index = Symb.p & ~uintptr_t(3);
  Index >>= 2;
  return Index;
}

void IRObjectFile::moveSymbolNext(DataRefImpl &Symb) const {
  const GlobalValue *GV = getGV(Symb);
  uintptr_t Res;

  switch (Symb.p & 3) {
  case 0: {
    Module::const_iterator Iter(static_cast<const Function*>(GV));
    ++Iter;
    Res = skipEmpty(Iter, *M);
    break;
  }
  case 1: {
    Module::const_global_iterator Iter(static_cast<const GlobalVariable*>(GV));
    ++Iter;
    Res = skipEmpty(Iter, *M);
    break;
  }
  case 2: {
    Module::const_alias_iterator Iter(static_cast<const GlobalAlias*>(GV));
    ++Iter;
    Res = skipEmpty(Iter, *M);
    break;
  }
  case 3: {
    unsigned Index = getAsmSymIndex(Symb);
    assert(Index < AsmSymbols.size());
    ++Index;
    Res = (Index << 2) | 3;
    break;
  }
  }

  Symb.p = Res;
}

std::error_code IRObjectFile::printSymbolName(raw_ostream &OS,
                                              DataRefImpl Symb) const {
  const GlobalValue *GV = getGV(Symb);
  if (!GV) {
    unsigned Index = getAsmSymIndex(Symb);
    assert(Index <= AsmSymbols.size());
    OS << AsmSymbols[Index].first;
    return object_error::success;;
  }

  if (Mang)
    Mang->getNameWithPrefix(OS, GV, false);
  else
    OS << GV->getName();

  return object_error::success;
}

static bool isDeclaration(const GlobalValue &V) {
  if (V.hasAvailableExternallyLinkage())
    return true;

  if (V.isMaterializable())
    return false;

  return V.isDeclaration();
}

uint32_t IRObjectFile::getSymbolFlags(DataRefImpl Symb) const {
  const GlobalValue *GV = getGV(Symb);

  if (!GV) {
    unsigned Index = getAsmSymIndex(Symb);
    assert(Index <= AsmSymbols.size());
    return AsmSymbols[Index].second;
  }

  uint32_t Res = BasicSymbolRef::SF_None;
  if (isDeclaration(*GV))
    Res |= BasicSymbolRef::SF_Undefined;
  if (GV->hasPrivateLinkage())
    Res |= BasicSymbolRef::SF_FormatSpecific;
  if (!GV->hasLocalLinkage())
    Res |= BasicSymbolRef::SF_Global;
  if (GV->hasCommonLinkage())
    Res |= BasicSymbolRef::SF_Common;
  if (GV->hasLinkOnceLinkage() || GV->hasWeakLinkage())
    Res |= BasicSymbolRef::SF_Weak;

  if (GV->getName().startswith("llvm."))
    Res |= BasicSymbolRef::SF_FormatSpecific;
  else if (auto *Var = dyn_cast<GlobalVariable>(GV)) {
    if (Var->getSection() == StringRef("llvm.metadata"))
      Res |= BasicSymbolRef::SF_FormatSpecific;
  }

  return Res;
}

const GlobalValue *IRObjectFile::getSymbolGV(DataRefImpl Symb) const {
  const GlobalValue *GV = getGV(Symb);
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
  uint64_t NumAsm = AsmSymbols.size();
  NumAsm <<= 2;
  Ret.p = 3 | NumAsm;
  return basic_symbol_iterator(BasicSymbolRef(Ret, this));
}

ErrorOr<IRObjectFile *> llvm::object::IRObjectFile::createIRObjectFile(
    std::unique_ptr<MemoryBuffer> Object, LLVMContext &Context) {
  ErrorOr<Module *> MOrErr = getLazyBitcodeModule(Object.get(), Context);
  if (std::error_code EC = MOrErr.getError())
    return EC;

  std::unique_ptr<Module> M(MOrErr.get());
  return new IRObjectFile(std::move(Object), std::move(M));
}
