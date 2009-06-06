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
  const TargetData *TD = TM.getTargetData();
  const Function *F = MF.getFunction();

  // Align the output buffer to the appropriate alignment, power of 2.
  unsigned FnAlign = F->getAlignment();
  unsigned TDAlign = TD->getPrefTypeAlignment(F->getType());
  unsigned Align = std::max(FnAlign, TDAlign);
  assert(!(Align & (Align-1)) && "Alignment is not a power of two!");

  // Get the ELF Section that this function belongs in.
  ES = &EW.getTextSection();

  // FIXME: better memory management, this will be replaced by BinaryObjects
  ES->SectionData.reserve(4096);
  BufferBegin = &ES->SectionData[0];
  BufferEnd = BufferBegin + ES->SectionData.capacity();

  // Upgrade the section alignment if required.
  if (ES->Align < Align) ES->Align = Align;

  // Round the size up to the correct alignment for starting the new function.
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

  // Figure out the binding (linkage) of the symbol.
  switch (MF.getFunction()->getLinkage()) {
  default:
    // appending linkage is illegal for functions.
    assert(0 && "Unknown linkage type!");
  case GlobalValue::ExternalLinkage:
    FnSym.SetBind(ELFSym::STB_GLOBAL);
    break;
  case GlobalValue::LinkOnceAnyLinkage:
  case GlobalValue::LinkOnceODRLinkage:
  case GlobalValue::WeakAnyLinkage:
  case GlobalValue::WeakODRLinkage:
    FnSym.SetBind(ELFSym::STB_WEAK);
    break;
  case GlobalValue::PrivateLinkage:
    assert (0 && "PrivateLinkage should not be in the symbol table.");
  case GlobalValue::InternalLinkage:
    FnSym.SetBind(ELFSym::STB_LOCAL);
    break;
  }

  // Set the symbol type as a function
  FnSym.SetType(ELFSym::STT_FUNC);

  FnSym.SectionIdx = ES->SectionIdx;
  FnSym.Size = CurBufferPtr-FnStartPtr;

  // Offset from start of Section
  FnSym.Value = FnStartPtr-BufferBegin;

  // Finally, add it to the symtab.
  EW.SymbolTable.push_back(FnSym);

  // Update Section Size
  ES->Size = CurBufferPtr - BufferBegin;
  return false;
}

} // end namespace llvm
