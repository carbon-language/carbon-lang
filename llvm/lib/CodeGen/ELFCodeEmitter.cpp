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
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetAsmInfo.h"
#include "llvm/Support/Debug.h"

//===----------------------------------------------------------------------===//
//                       ELFCodeEmitter Implementation
//===----------------------------------------------------------------------===//

namespace llvm {

/// startFunction - This callback is invoked when a new machine function is
/// about to be emitted.
void ELFCodeEmitter::startFunction(MachineFunction &MF) {
  // Get the ELF Section that this function belongs in.
  ES = &EW.getTextSection();

  DOUT << "processing function: " << MF.getFunction()->getName() << "\n";

  // FIXME: better memory management, this will be replaced by BinaryObjects
  BinaryData &BD = ES->getData();
  BD.reserve(4096);
  BufferBegin = &BD[0];
  BufferEnd = BufferBegin + BD.capacity();

  // Get the function alignment in bytes
  unsigned Align = (1 << MF.getAlignment());

  // Align the section size with the function alignment, so the function can
  // start in a aligned offset, also update the section alignment if needed.
  if (ES->Align < Align) ES->Align = Align;
  ES->Size = (ES->Size + (Align-1)) & (-Align);

  // Snaity check on allocated space for text section
  assert( ES->Size < 4096 && "no more space in TextSection" );

  // FIXME: Using ES->Size directly here instead of calculating it from the
  // output buffer size (impossible because the code emitter deals only in raw
  // bytes) forces us to manually synchronize size and write padding zero bytes
  // to the output buffer for all non-text sections.  For text sections, we do
  // not synchonize the output buffer, and we just blow up if anyone tries to
  // write non-code to it.  An assert should probably be added to
  // AddSymbolToSection to prevent calling it on the text section.
  CurBufferPtr = BufferBegin + ES->Size;

  // Record function start address relative to BufferBegin
  FnStartPtr = CurBufferPtr;
}

/// finishFunction - This callback is invoked after the function is completely
/// finished.
bool ELFCodeEmitter::finishFunction(MachineFunction &MF) {
  // Update Section Size
  ES->Size = CurBufferPtr - BufferBegin;

  // Add a symbol to represent the function.
  const Function *F = MF.getFunction();
  ELFSym FnSym(F);
  FnSym.setType(ELFSym::STT_FUNC);
  FnSym.setBind(EW.getGlobalELFLinkage(F));
  FnSym.setVisibility(EW.getGlobalELFVisibility(F));
  FnSym.SectionIdx = ES->SectionIdx;
  FnSym.Size = CurBufferPtr-FnStartPtr;

  // Offset from start of Section
  FnSym.Value = FnStartPtr-BufferBegin;

  // Locals should go on the symbol list front
  if (!F->hasPrivateLinkage()) {
    if (FnSym.getBind() == ELFSym::STB_LOCAL)
      EW.SymbolList.push_front(FnSym);
    else
      EW.SymbolList.push_back(FnSym);
  }

  // Emit constant pool to appropriate section(s)
  emitConstantPool(MF.getConstantPool());

  // Emit jump tables to appropriate section
  emitJumpTables(MF.getJumpTableInfo());

  // Relocations
  // -----------
  // If we have emitted any relocations to function-specific objects such as
  // basic blocks, constant pools entries, or jump tables, record their
  // addresses now so that we can rewrite them with the correct addresses
  // later.
  for (unsigned i = 0, e = Relocations.size(); i != e; ++i) {
    MachineRelocation &MR = Relocations[i];
    intptr_t Addr;
    if (MR.isGlobalValue()) {
      EW.PendingGlobals.insert(MR.getGlobalValue());
    } else if (MR.isBasicBlock()) {
      Addr = getMachineBasicBlockAddress(MR.getBasicBlock());
      MR.setConstantVal(ES->SectionIdx);
      MR.setResultPointer((void*)Addr);
    } else if (MR.isConstantPoolIndex()) {
      Addr = getConstantPoolEntryAddress(MR.getConstantPoolIndex());
      MR.setConstantVal(CPSections[MR.getConstantPoolIndex()]);
      MR.setResultPointer((void*)Addr);
    } else if (MR.isJumpTableIndex()) {
      Addr = getJumpTableEntryAddress(MR.getJumpTableIndex());
      MR.setResultPointer((void*)Addr);
      MR.setConstantVal(JumpTableSectionIdx);
    } else {
      assert(0 && "Unhandled relocation type");
    }
    ES->addRelocation(MR);
  }

  // Clear per-function data structures.
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

  const TargetAsmInfo *TAI = TM.getTargetAsmInfo();
  for (unsigned i = 0, e = CP.size(); i != e; ++i) {
    MachineConstantPoolEntry CPE = CP[i];

    // Get the right ELF Section for this constant pool entry
    std::string CstPoolName =
      TAI->SelectSectionForMachineConst(CPE.getType())->getName();
    ELFSection &CstPoolSection =
      EW.getConstantPoolSection(CstPoolName, CPE.getAlignment());

    // Record the constant pool location and the section index
    CPLocations.push_back(CstPoolSection.size());
    CPSections.push_back(CstPoolSection.SectionIdx);

    if (CPE.isMachineConstantPoolEntry())
      assert("CPE.isMachineConstantPoolEntry not supported yet");

    // Emit the constant to constant pool section
    EW.EmitGlobalConstant(CPE.Val.ConstVal, CstPoolSection);
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

  const TargetAsmInfo *TAI = TM.getTargetAsmInfo();

  // Get the ELF Section to emit the jump table
  unsigned Align = TM.getTargetData()->getPointerABIAlignment();
  std::string JTName(TAI->getJumpTableDataSection());
  ELFSection &JTSection = EW.getJumpTableSection(JTName, Align);
  JumpTableSectionIdx = JTSection.SectionIdx;

  // Entries in the JT Section are relocated against the text section
  ELFSection &TextSection = EW.getTextSection();

  // For each JT, record its offset from the start of the section
  for (unsigned i = 0, e = JT.size(); i != e; ++i) {
    const std::vector<MachineBasicBlock*> &MBBs = JT[i].MBBs;

    DOUT << "JTSection.size(): " << JTSection.size() << "\n";
    DOUT << "JTLocations.size: " << JTLocations.size() << "\n";

    // Record JT 'i' offset in the JT section
    JTLocations.push_back(JTSection.size());

    // Each MBB entry in the Jump table section has a relocation entry
    // against the current text section.
    for (unsigned mi = 0, me = MBBs.size(); mi != me; ++mi) {
      MachineRelocation MR =
        MachineRelocation::getBB(JTSection.size(),
                                 MachineRelocation::VANILLA,
                                 MBBs[mi]);

      // Offset of JT 'i' in JT section
      MR.setResultPointer((void*)getMachineBasicBlockAddress(MBBs[mi]));
      MR.setConstantVal(TextSection.SectionIdx);

      // Add the relocation to the Jump Table section
      JTSection.addRelocation(MR);

      // Output placeholder for MBB in the JT section
      JTSection.emitWord(0);
    }
  }
}

} // end namespace llvm
