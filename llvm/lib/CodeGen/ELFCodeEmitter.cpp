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

  // Relocations
  // -----------
  // If we have emitted any relocations to function-specific objects such as 
  // basic blocks, constant pools entries, or jump tables, record their
  // addresses now so that we can rewrite them with the correct addresses
  // later.
  for (unsigned i = 0, e = Relocations.size(); i != e; ++i) {
    MachineRelocation &MR = Relocations[i];
    intptr_t Addr;
    if (MR.isBasicBlock()) {
      Addr = getMachineBasicBlockAddress(MR.getBasicBlock());
      MR.setConstantVal(ES->SectionIdx);
      MR.setResultPointer((void*)Addr);
    } else if (MR.isGlobalValue()) {
      EW.PendingGlobals.insert(MR.getGlobalValue());
    } else {
      assert(0 && "Unhandled relocation type");
    }
    ES->addRelocation(MR);
  }
  Relocations.clear();

  return false;
}

} // end namespace llvm
