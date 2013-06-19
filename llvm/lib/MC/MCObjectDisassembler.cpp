//===- lib/MC/MCObjectDisassembler.cpp ------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/MC/MCObjectDisassembler.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/MC/MCAtom.h"
#include "llvm/MC/MCDisassembler.h"
#include "llvm/MC/MCFunction.h"
#include "llvm/MC/MCInstrAnalysis.h"
#include "llvm/MC/MCModule.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/MemoryObject.h"
#include "llvm/Support/StringRefMemoryObject.h"
#include "llvm/Support/raw_ostream.h"
#include <map>
#include <set>

using namespace llvm;
using namespace object;

MCObjectDisassembler::MCObjectDisassembler(const ObjectFile &Obj,
                                           const MCDisassembler &Dis,
                                           const MCInstrAnalysis &MIA)
  : Obj(Obj), Dis(Dis), MIA(MIA) {}

MCModule *MCObjectDisassembler::buildModule(bool withCFG) {
  MCModule *Module = new MCModule;
  buildSectionAtoms(Module);
  if (withCFG)
    buildCFG(Module);
  return Module;
}

void MCObjectDisassembler::buildSectionAtoms(MCModule *Module) {
  error_code ec;
  for (section_iterator SI = Obj.begin_sections(),
                        SE = Obj.end_sections();
                        SI != SE;
                        SI.increment(ec)) {
    if (ec) break;

    bool isText; SI->isText(isText);
    bool isData; SI->isData(isData);
    if (!isData && !isText)
      continue;

    uint64_t StartAddr; SI->getAddress(StartAddr);
    uint64_t SecSize; SI->getSize(SecSize);
    if (StartAddr == UnknownAddressOrSize || SecSize == UnknownAddressOrSize)
      continue;

    StringRef Contents; SI->getContents(Contents);
    StringRefMemoryObject memoryObject(Contents);

    // We don't care about things like non-file-backed sections yet.
    if (Contents.size() != SecSize || !SecSize)
      continue;
    uint64_t EndAddr = StartAddr + SecSize - 1;

    StringRef SecName; SI->getName(SecName);

    if (isText) {
      MCTextAtom *Text = Module->createTextAtom(StartAddr, EndAddr);
      Text->setName(SecName);
      uint64_t InstSize;
      for (uint64_t Index = 0; Index < SecSize; Index += InstSize) {
        MCInst Inst;
        if (Dis.getInstruction(Inst, InstSize, memoryObject, Index,
                               nulls(), nulls()))
          Text->addInst(Inst, InstSize);
        else
          // We don't care about splitting mixed atoms either.
          llvm_unreachable("Couldn't disassemble instruction in atom.");
      }

    } else {
      MCDataAtom *Data = Module->createDataAtom(StartAddr, EndAddr);
      Data->setName(SecName);
      for (uint64_t Index = 0; Index < SecSize; ++Index)
        Data->addData(Contents[Index]);
    }
  }
}

namespace {
  struct BBInfo;
  typedef std::set<BBInfo*> BBInfoSetTy;

  struct BBInfo {
    MCTextAtom *Atom;
    MCBasicBlock *BB;
    BBInfoSetTy Succs;
    BBInfoSetTy Preds;

    void addSucc(BBInfo &Succ) {
      Succs.insert(&Succ);
      Succ.Preds.insert(this);
    }
  };
}

void MCObjectDisassembler::buildCFG(MCModule *Module) {
  typedef std::map<uint64_t, BBInfo> BBInfoByAddrTy;
  BBInfoByAddrTy BBInfos;
  typedef std::set<uint64_t> AddressSetTy;
  AddressSetTy Splits;
  AddressSetTy Calls;

  assert(Module->func_begin() == Module->func_end()
         && "Module already has a CFG!");

  // First, determine the basic block boundaries and call targets.
  for (MCModule::atom_iterator AI = Module->atom_begin(),
                               AE = Module->atom_end();
       AI != AE; ++AI) {
    MCTextAtom *TA = dyn_cast<MCTextAtom>(*AI);
    if (!TA) continue;
    Calls.insert(TA->getBeginAddr());
    BBInfos[TA->getBeginAddr()].Atom = TA;
    for (MCTextAtom::const_iterator II = TA->begin(), IE = TA->end();
         II != IE; ++II) {
      if (MIA.isTerminator(II->Inst))
        Splits.insert(II->Address + II->Size);
      uint64_t Target;
      if (MIA.evaluateBranch(II->Inst, II->Address, II->Size, Target)) {
        if (MIA.isCall(II->Inst))
          Calls.insert(Target);
        Splits.insert(Target);
      }
    }
  }

  // Split text atoms into basic block atoms.
  for (AddressSetTy::const_iterator SI = Splits.begin(), SE = Splits.end();
       SI != SE; ++SI) {
    MCAtom *A = Module->findAtomContaining(*SI);
    if (!A) continue;
    MCTextAtom *TA = cast<MCTextAtom>(A);
    if (TA->getBeginAddr() == *SI)
      continue;
    MCTextAtom *NewAtom = TA->split(*SI);
    BBInfos[NewAtom->getBeginAddr()].Atom = NewAtom;
    StringRef BBName = TA->getName();
    BBName = BBName.substr(0, BBName.find_last_of(':'));
    NewAtom->setName((BBName + ":" + utohexstr(*SI)).str());
  }

  // Compute succs/preds.
  for (MCModule::atom_iterator AI = Module->atom_begin(),
                               AE = Module->atom_end();
                               AI != AE; ++AI) {
    MCTextAtom *TA = dyn_cast<MCTextAtom>(*AI);
    if (!TA) continue;
    BBInfo &CurBB = BBInfos[TA->getBeginAddr()];
    const MCDecodedInst &LI = TA->back();
    if (MIA.isBranch(LI.Inst)) {
      uint64_t Target;
      if (MIA.evaluateBranch(LI.Inst, LI.Address, LI.Size, Target))
        CurBB.addSucc(BBInfos[Target]);
      if (MIA.isConditionalBranch(LI.Inst))
        CurBB.addSucc(BBInfos[LI.Address + LI.Size]);
    } else if (!MIA.isTerminator(LI.Inst))
      CurBB.addSucc(BBInfos[LI.Address + LI.Size]);
  }


  // Create functions and basic blocks.
  for (AddressSetTy::const_iterator CI = Calls.begin(), CE = Calls.end();
       CI != CE; ++CI) {
    BBInfo &BBI = BBInfos[*CI];
    if (!BBI.Atom) continue;

    MCFunction &MCFN = *Module->createFunction(BBI.Atom->getName());

    // Create MCBBs.
    SmallSetVector<BBInfo*, 16> Worklist;
    Worklist.insert(&BBI);
    for (size_t WI = 0; WI < Worklist.size(); ++WI) {
      BBInfo *BBI = Worklist[WI];
      if (!BBI->Atom)
        continue;
      BBI->BB = &MCFN.createBlock(*BBI->Atom);
      // Add all predecessors and successors to the worklist.
      for (BBInfoSetTy::iterator SI = BBI->Succs.begin(), SE = BBI->Succs.end();
                                 SI != SE; ++SI)
        Worklist.insert(*SI);
      for (BBInfoSetTy::iterator PI = BBI->Preds.begin(), PE = BBI->Preds.end();
                                 PI != PE; ++PI)
        Worklist.insert(*PI);
    }

    // Set preds/succs.
    for (size_t WI = 0; WI < Worklist.size(); ++WI) {
      BBInfo *BBI = Worklist[WI];
      MCBasicBlock *MCBB = BBI->BB;
      if (!MCBB)
        continue;
      for (BBInfoSetTy::iterator SI = BBI->Succs.begin(), SE = BBI->Succs.end();
                                 SI != SE; ++SI)
        MCBB->addSuccessor((*SI)->BB);
      for (BBInfoSetTy::iterator PI = BBI->Preds.begin(), PE = BBI->Preds.end();
                                 PI != PE; ++PI)
        MCBB->addPredecessor((*PI)->BB);
    }
  }
}
