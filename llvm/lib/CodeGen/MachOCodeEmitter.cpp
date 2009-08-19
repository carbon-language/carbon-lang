//===-- MachOEmitter.cpp - Target-independent Mach-O Emitter code --------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "MachO.h"
#include "MachOWriter.h"
#include "MachOCodeEmitter.h"
#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Function.h"
#include "llvm/CodeGen/MachineConstantPool.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineJumpTableInfo.h"
#include "llvm/CodeGen/MachineRelocation.h"
#include "llvm/Target/TargetAsmInfo.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/Mangler.h"
#include "llvm/Support/OutputBuffer.h"
#include <vector>

//===----------------------------------------------------------------------===//
//                       MachOCodeEmitter Implementation
//===----------------------------------------------------------------------===//

namespace llvm {

MachOCodeEmitter::MachOCodeEmitter(MachOWriter &mow, MachOSection &mos) :
      ObjectCodeEmitter(&mos), MOW(mow), TM(MOW.TM) {
  is64Bit = TM.getTargetData()->getPointerSizeInBits() == 64;
  isLittleEndian = TM.getTargetData()->isLittleEndian();
  TAI = TM.getTargetAsmInfo();
}

/// startFunction - This callback is invoked when a new machine function is
/// about to be emitted.

void MachOCodeEmitter::startFunction(MachineFunction &MF) {
  const TargetData *TD = TM.getTargetData();
  const Function *F = MF.getFunction();

  // Align the output buffer to the appropriate alignment, power of 2.
  unsigned FnAlign = F->getAlignment();
  unsigned TDAlign = TD->getPrefTypeAlignment(F->getType());
  unsigned Align = Log2_32(std::max(FnAlign, TDAlign));
  assert(!(Align & (Align-1)) && "Alignment is not a power of two!");

  // Get the Mach-O Section that this function belongs in.
  MachOSection *MOS = MOW.getTextSection();
  
  // Upgrade the section alignment if required.
  if (MOS->align < Align) MOS->align = Align;

  MOS->emitAlignment(Align);

  // Create symbol for function entry
  const GlobalValue *FuncV = MF.getFunction();
  MachOSym FnSym(FuncV, MOW.Mang->getMangledName(FuncV), MOS->Index, TAI);
  FnSym.n_value = getCurrentPCOffset();

  // add it to the symtab.
  MOW.SymbolTable.push_back(FnSym);
}

/// finishFunction - This callback is invoked after the function is completely
/// finished.

bool MachOCodeEmitter::finishFunction(MachineFunction &MF) {
    
  // Get the Mach-O Section that this function belongs in.
  MachOSection *MOS = MOW.getTextSection();

  // Emit constant pool to appropriate section(s)
  emitConstantPool(MF.getConstantPool());

  // Emit jump tables to appropriate section
  emitJumpTables(MF.getJumpTableInfo());
  
  // If we have emitted any relocations to function-specific objects such as 
  // basic blocks, constant pools entries, or jump tables, record their
  // addresses now so that we can rewrite them with the correct addresses
  // later.
  for (unsigned i = 0, e = Relocations.size(); i != e; ++i) {
    MachineRelocation &MR = Relocations[i];
    intptr_t Addr;

    if (MR.isBasicBlock()) {
      Addr = getMachineBasicBlockAddress(MR.getBasicBlock());
      MR.setConstantVal(MOS->Index);
      MR.setResultPointer((void*)Addr);
    } else if (MR.isJumpTableIndex()) {
      Addr = getJumpTableEntryAddress(MR.getJumpTableIndex());
      MR.setConstantVal(MOW.getJumpTableSection()->Index);
      MR.setResultPointer((void*)Addr);
    } else if (MR.isConstantPoolIndex()) {
      Addr = getConstantPoolEntryAddress(MR.getConstantPoolIndex());
      MR.setConstantVal(CPSections[MR.getConstantPoolIndex()]);
      MR.setResultPointer((void*)Addr);
    } else if (MR.isGlobalValue()) {
      // FIXME: This should be a set or something that uniques
      MOW.PendingGlobals.push_back(MR.getGlobalValue());
    } else {
      llvm_unreachable("Unhandled relocation type");
    }
    MOS->addRelocation(MR);
  }
  Relocations.clear();

  // Clear per-function data structures.
  CPLocations.clear();
  CPSections.clear();
  JTLocations.clear();
  MBBLocations.clear();

  return false;
}

/// emitConstantPool - For each constant pool entry, figure out which section
/// the constant should live in, allocate space for it, and emit it to the 
/// Section data buffer.
void MachOCodeEmitter::emitConstantPool(MachineConstantPool *MCP) {
  const std::vector<MachineConstantPoolEntry> &CP = MCP->getConstants();
  if (CP.empty()) return;

  // FIXME: handle PIC codegen
  assert(TM.getRelocationModel() != Reloc::PIC_ &&
         "PIC codegen not yet handled for mach-o jump tables!");

  // Although there is no strict necessity that I am aware of, we will do what
  // gcc for OS X does and put each constant pool entry in a section of constant
  // objects of a certain size.  That means that float constants go in the
  // literal4 section, and double objects go in literal8, etc.
  //
  // FIXME: revisit this decision if we ever do the "stick everything into one
  // "giant object for PIC" optimization.
  for (unsigned i = 0, e = CP.size(); i != e; ++i) {
    const Type *Ty = CP[i].getType();
    unsigned Size = TM.getTargetData()->getTypeAllocSize(Ty);

    MachOSection *Sec = MOW.getConstSection(CP[i].Val.ConstVal);
    OutputBuffer SecDataOut(Sec->getData(), is64Bit, isLittleEndian);

    CPLocations.push_back(Sec->size());
    CPSections.push_back(Sec->Index);

    // Allocate space in the section for the global.
    // FIXME: need alignment?
    // FIXME: share between here and AddSymbolToSection?
    for (unsigned j = 0; j < Size; ++j)
      SecDataOut.outbyte(0);

    MachOWriter::InitMem(CP[i].Val.ConstVal, CPLocations[i],
                         TM.getTargetData(), Sec);
  }
}

/// emitJumpTables - Emit all the jump tables for a given jump table info
/// record to the appropriate section.
void MachOCodeEmitter::emitJumpTables(MachineJumpTableInfo *MJTI) {
  const std::vector<MachineJumpTableEntry> &JT = MJTI->getJumpTables();
  if (JT.empty()) return;

  // FIXME: handle PIC codegen
  assert(TM.getRelocationModel() != Reloc::PIC_ &&
         "PIC codegen not yet handled for mach-o jump tables!");

  MachOSection *Sec = MOW.getJumpTableSection();
  unsigned TextSecIndex = MOW.getTextSection()->Index;
  OutputBuffer SecDataOut(Sec->getData(), is64Bit, isLittleEndian);

  for (unsigned i = 0, e = JT.size(); i != e; ++i) {
    // For each jump table, record its offset from the start of the section,
    // reserve space for the relocations to the MBBs, and add the relocations.
    const std::vector<MachineBasicBlock*> &MBBs = JT[i].MBBs;
    JTLocations.push_back(Sec->size());
    for (unsigned mi = 0, me = MBBs.size(); mi != me; ++mi) {
      MachineRelocation MR(MOW.GetJTRelocation(Sec->size(), MBBs[mi]));
      MR.setResultPointer((void *)JTLocations[i]);
      MR.setConstantVal(TextSecIndex);
      Sec->addRelocation(MR);
      SecDataOut.outaddr(0);
    }
  }
}

} // end namespace llvm

