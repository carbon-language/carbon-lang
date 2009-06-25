//===-- lib/CodeGen/ELFCodeEmitter.cpp ------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "elfce"

#include "ELFCodeEmitter.h"
#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Function.h"
#include "llvm/CodeGen/BinaryObject.h"
#include "llvm/CodeGen/MachineConstantPool.h"
#include "llvm/CodeGen/MachineJumpTableInfo.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Target/TargetMachine.h"
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

  // Align the output buffer with function alignment, and
  // upgrade the section alignment if required
  unsigned Align =
    TM.getELFWriterInfo()->getFunctionAlignment(MF.getFunction());
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
  // Add a symbol to represent the function.
  ELFSym FnSym(MF.getFunction());

  // Update Section Size
  ES->Size = CurBufferPtr - BufferBegin;

  // Set the symbol type as a function
  FnSym.setType(ELFSym::STT_FUNC);
  FnSym.SectionIdx = ES->SectionIdx;
  FnSym.Size = CurBufferPtr-FnStartPtr;

  // Offset from start of Section
  FnSym.Value = FnStartPtr-BufferBegin;

  // Figure out the binding (linkage) of the symbol.
  switch (MF.getFunction()->getLinkage()) {
  default:
    // appending linkage is illegal for functions.
    assert(0 && "Unknown linkage type!");
  case GlobalValue::ExternalLinkage:
    FnSym.setBind(ELFSym::STB_GLOBAL);
    EW.SymbolList.push_back(FnSym);
    break;
  case GlobalValue::LinkOnceAnyLinkage:
  case GlobalValue::LinkOnceODRLinkage:
  case GlobalValue::WeakAnyLinkage:
  case GlobalValue::WeakODRLinkage:
    FnSym.setBind(ELFSym::STB_WEAK);
    EW.SymbolList.push_back(FnSym);
    break;
  case GlobalValue::PrivateLinkage:
    assert (0 && "PrivateLinkage should not be in the symbol table.");
  case GlobalValue::InternalLinkage:
    FnSym.setBind(ELFSym::STB_LOCAL);
    EW.SymbolList.push_front(FnSym);
    break;
  }

  // Emit constant pool to appropriate section(s)
  emitConstantPool(MF.getConstantPool());

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
    } else {
      assert(0 && "Unhandled relocation type");
    }
    ES->addRelocation(MR);
  }
  Relocations.clear();

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

} // end namespace llvm
