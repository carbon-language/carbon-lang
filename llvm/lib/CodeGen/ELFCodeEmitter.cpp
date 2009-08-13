//===-- lib/CodeGen/ELFCodeEmitter.cpp ------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "elfce"

#include "ELF.h"
#include "ELFWriter.h"
#include "ELFCodeEmitter.h"
#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Function.h"
#include "llvm/CodeGen/BinaryObject.h"
#include "llvm/CodeGen/MachineConstantPool.h"
#include "llvm/CodeGen/MachineJumpTableInfo.h"
#include "llvm/CodeGen/MachineRelocation.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Target/TargetELFWriterInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetAsmInfo.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

//===----------------------------------------------------------------------===//
//                       ELFCodeEmitter Implementation
//===----------------------------------------------------------------------===//

namespace llvm {

/// startFunction - This callback is invoked when a new machine function is
/// about to be emitted.
void ELFCodeEmitter::startFunction(MachineFunction &MF) {
  DEBUG(errs() << "processing function: "
        << MF.getFunction()->getName() << "\n");

  // Get the ELF Section that this function belongs in.
  ES = &EW.getTextSection(MF.getFunction());

  // Set the desired binary object to be used by the code emitters
  setBinaryObject(ES);

  // Get the function alignment in bytes
  unsigned Align = (1 << MF.getAlignment());

  // The function must start on its required alignment
  ES->emitAlignment(Align);

  // Update the section alignment if needed.
  ES->Align = std::max(ES->Align, Align);

  // Record the function start offset
  FnStartOff = ES->getCurrentPCOffset();

  // Emit constant pool and jump tables to their appropriate sections.
  // They need to be emitted before the function because in some targets
  // the later may reference JT or CP entry address.
  emitConstantPool(MF.getConstantPool());
  emitJumpTables(MF.getJumpTableInfo());
}

/// finishFunction - This callback is invoked after the function is completely
/// finished.
bool ELFCodeEmitter::finishFunction(MachineFunction &MF) {
  // Add a symbol to represent the function.
  const Function *F = MF.getFunction();
  ELFSym *FnSym = ELFSym::getGV(F, EW.getGlobalELFBinding(F), ELFSym::STT_FUNC,
                                EW.getGlobalELFVisibility(F));
  FnSym->SectionIdx = ES->SectionIdx;
  FnSym->Size = ES->getCurrentPCOffset()-FnStartOff;
  EW.AddPendingGlobalSymbol(F, true);

  // Offset from start of Section
  FnSym->Value = FnStartOff;

  if (!F->hasPrivateLinkage())
    EW.SymbolList.push_back(FnSym);

  // Patch up Jump Table Section relocations to use the real MBBs offsets
  // now that the MBB label offsets inside the function are known.
  if (!MF.getJumpTableInfo()->isEmpty()) {
    ELFSection &JTSection = EW.getJumpTableSection();
    for (std::vector<MachineRelocation>::iterator MRI = JTRelocations.begin(),
         MRE = JTRelocations.end(); MRI != MRE; ++MRI) {
      MachineRelocation &MR = *MRI;
      unsigned MBBOffset = getMachineBasicBlockAddress(MR.getBasicBlock());
      MR.setResultPointer((void*)MBBOffset);
      MR.setConstantVal(ES->SectionIdx);
      JTSection.addRelocation(MR);
    }
  }

  // If we have emitted any relocations to function-specific objects such as
  // basic blocks, constant pools entries, or jump tables, record their
  // addresses now so that we can rewrite them with the correct addresses later
  for (unsigned i = 0, e = Relocations.size(); i != e; ++i) {
    MachineRelocation &MR = Relocations[i];
    intptr_t Addr;
    if (MR.isGlobalValue()) {
      EW.AddPendingGlobalSymbol(MR.getGlobalValue());
    } else if (MR.isExternalSymbol()) {
      EW.AddPendingExternalSymbol(MR.getExternalSymbol());
    } else if (MR.isBasicBlock()) {
      Addr = getMachineBasicBlockAddress(MR.getBasicBlock());
      MR.setConstantVal(ES->SectionIdx);
      MR.setResultPointer((void*)Addr);
    } else if (MR.isConstantPoolIndex()) {
      Addr = getConstantPoolEntryAddress(MR.getConstantPoolIndex());
      MR.setConstantVal(CPSections[MR.getConstantPoolIndex()]);
      MR.setResultPointer((void*)Addr);
    } else if (MR.isJumpTableIndex()) {
      ELFSection &JTSection = EW.getJumpTableSection();
      Addr = getJumpTableEntryAddress(MR.getJumpTableIndex());
      MR.setConstantVal(JTSection.SectionIdx);
      MR.setResultPointer((void*)Addr);
    } else {
      llvm_unreachable("Unhandled relocation type");
    }
    ES->addRelocation(MR);
  }

  // Clear per-function data structures.
  JTRelocations.clear();
  Relocations.clear();
  CPLocations.clear();
  CPSections.clear();
  JTLocations.clear();
  MBBLocations.clear();
  return false;
}

/// emitConstantPool - For each constant pool entry, figure out which section
/// the constant should live in and emit the constant
void ELFCodeEmitter::emitConstantPool(MachineConstantPool *MCP) {
  const std::vector<MachineConstantPoolEntry> &CP = MCP->getConstants();
  if (CP.empty()) return;

  // TODO: handle PIC codegen
  assert(TM.getRelocationModel() != Reloc::PIC_ &&
         "PIC codegen not yet handled for elf constant pools!");

  for (unsigned i = 0, e = CP.size(); i != e; ++i) {
    MachineConstantPoolEntry CPE = CP[i];

    // Record the constant pool location and the section index
    ELFSection &CstPool = EW.getConstantPoolSection(CPE);
    CPLocations.push_back(CstPool.size());
    CPSections.push_back(CstPool.SectionIdx);

    if (CPE.isMachineConstantPoolEntry())
      assert("CPE.isMachineConstantPoolEntry not supported yet");

    // Emit the constant to constant pool section
    EW.EmitGlobalConstant(CPE.Val.ConstVal, CstPool);
  }
}

/// emitJumpTables - Emit all the jump tables for a given jump table info
/// record to the appropriate section.
void ELFCodeEmitter::emitJumpTables(MachineJumpTableInfo *MJTI) {
  const std::vector<MachineJumpTableEntry> &JT = MJTI->getJumpTables();
  if (JT.empty()) return;

  // FIXME: handle PIC codegen
  assert(TM.getRelocationModel() != Reloc::PIC_ &&
         "PIC codegen not yet handled for elf jump tables!");

  const TargetELFWriterInfo *TEW = TM.getELFWriterInfo();
  unsigned EntrySize = MJTI->getEntrySize();

  // Get the ELF Section to emit the jump table
  ELFSection &JTSection = EW.getJumpTableSection();

  // For each JT, record its offset from the start of the section
  for (unsigned i = 0, e = JT.size(); i != e; ++i) {
    const std::vector<MachineBasicBlock*> &MBBs = JT[i].MBBs;

    // Record JT 'i' offset in the JT section
    JTLocations.push_back(JTSection.size());

    // Each MBB entry in the Jump table section has a relocation entry
    // against the current text section.
    for (unsigned mi = 0, me = MBBs.size(); mi != me; ++mi) {
      unsigned MachineRelTy = TEW->getAbsoluteLabelMachineRelTy();
      MachineRelocation MR =
        MachineRelocation::getBB(JTSection.size(), MachineRelTy, MBBs[mi]);

      // Add the relocation to the Jump Table section
      JTRelocations.push_back(MR);

      // Output placeholder for MBB in the JT section
      for (unsigned s=0; s < EntrySize; ++s)
        JTSection.emitByte(0);
    }
  }
}

} // end namespace llvm
