//===-- lib/CodeGen/ELFCodeEmitter.cpp ------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "ELFCodeEmitter.h"
#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Function.h"
#include "llvm/CodeGen/MachineConstantPool.h"
#include "llvm/CodeGen/MachineJumpTableInfo.h"
#include "llvm/Target/TargetAsmInfo.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Support/Mangler.h"
#include "llvm/Support/OutputBuffer.h"

//===----------------------------------------------------------------------===//
//                       ELFCodeEmitter Implementation
//===----------------------------------------------------------------------===//

namespace llvm {

/// startFunction - This callback is invoked when a new machine function is
/// about to be emitted.
void ELFCodeEmitter::startFunction(MachineFunction &F) {
  // Align the output buffer to the appropriate alignment.
  unsigned Align = 16;   // FIXME: GENERICIZE!!
  // Get the ELF Section that this function belongs in.
  ES = &EW.getSection(".text", ELFWriter::ELFSection::SHT_PROGBITS,
                      ELFWriter::ELFSection::SHF_EXECINSTR |
                      ELFWriter::ELFSection::SHF_ALLOC);
  OutBuffer = &ES->SectionData;
  cerr << "FIXME: This code needs to be updated for changes in the "
       << "CodeEmitter interfaces.  In particular, this should set "
       << "BufferBegin/BufferEnd/CurBufferPtr, not deal with OutBuffer!";
  abort();

  // Upgrade the section alignment if required.
  if (ES->Align < Align) ES->Align = Align;

  // Add padding zeros to the end of the buffer to make sure that the
  // function will start on the correct byte alignment within the section.
  OutputBuffer OB(*OutBuffer,
                  TM.getTargetData()->getPointerSizeInBits() == 64,
                  TM.getTargetData()->isLittleEndian());
  OB.align(Align);
  FnStart = OutBuffer->size();
}

/// finishFunction - This callback is invoked after the function is completely
/// finished.
bool ELFCodeEmitter::finishFunction(MachineFunction &F) {
  // We now know the size of the function, add a symbol to represent it.
  ELFWriter::ELFSym FnSym(F.getFunction());

  // Figure out the binding (linkage) of the symbol.
  switch (F.getFunction()->getLinkage()) {
  default:
    // appending linkage is illegal for functions.
    assert(0 && "Unknown linkage type!");
  case GlobalValue::ExternalLinkage:
    FnSym.SetBind(ELFWriter::ELFSym::STB_GLOBAL);
    break;
  case GlobalValue::LinkOnceAnyLinkage:
  case GlobalValue::LinkOnceODRLinkage:
  case GlobalValue::WeakAnyLinkage:
  case GlobalValue::WeakODRLinkage:
    FnSym.SetBind(ELFWriter::ELFSym::STB_WEAK);
    break;
  case GlobalValue::PrivateLinkage:
    assert (0 && "PrivateLinkage should not be in the symbol table.");
  case GlobalValue::InternalLinkage:
    FnSym.SetBind(ELFWriter::ELFSym::STB_LOCAL);
    break;
  }

  ES->Size = OutBuffer->size();

  FnSym.SetType(ELFWriter::ELFSym::STT_FUNC);
  FnSym.SectionIdx = ES->SectionIdx;
  FnSym.Value = FnStart;   // Value = Offset from start of Section.
  FnSym.Size = OutBuffer->size()-FnStart;

  // Finally, add it to the symtab.
  EW.SymbolTable.push_back(FnSym);
  return false;
}

} // end namespace llvm
