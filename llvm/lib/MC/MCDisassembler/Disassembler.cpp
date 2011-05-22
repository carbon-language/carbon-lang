//===-- lib/MC/Disassembler.cpp - Disassembler Public C Interface -*- C -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "Disassembler.h"
#include "llvm-c/Disassembler.h"

#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCDisassembler.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstPrinter.h"
#include "llvm/MC/MCContext.h"
#include "llvm/Target/TargetRegistry.h"
#include "llvm/Target/TargetAsmInfo.h"  // FIXME.
#include "llvm/Target/TargetMachine.h"  // FIXME.
#include "llvm/Target/TargetSelect.h"
#include "llvm/Support/MemoryObject.h"

namespace llvm {
class Target;
} // namespace llvm
using namespace llvm;

// LLVMCreateDisasm() creates a disassembler for the TripleName.  Symbolic
// disassembly is supported by passing a block of information in the DisInfo
// parameter and specifying the TagType and callback functions as described in
// the header llvm-c/Disassembler.h .  The pointer to the block and the 
// functions can all be passed as NULL.  If successful, this returns a
// disassembler context.  If not, it returns NULL.
//
LLVMDisasmContextRef LLVMCreateDisasm(const char *TripleName, void *DisInfo,
                                      int TagType, LLVMOpInfoCallback GetOpInfo,
                                      LLVMSymbolLookupCallback SymbolLookUp) {
  // Initialize targets and assembly printers/parsers.
  llvm::InitializeAllTargetInfos();
  // FIXME: We shouldn't need to initialize the Target(Machine)s.
  llvm::InitializeAllTargets();
  llvm::InitializeAllAsmPrinters();
  llvm::InitializeAllAsmParsers();
  llvm::InitializeAllDisassemblers();

  // Get the target.
  std::string Error;
  const Target *TheTarget = TargetRegistry::lookupTarget(TripleName, Error);
  assert(TheTarget && "Unable to create target!");

  // Get the assembler info needed to setup the MCContext.
  const MCAsmInfo *MAI = TheTarget->createAsmInfo(TripleName);
  assert(MAI && "Unable to create target asm info!");

  // Package up features to be passed to target/subtarget
  std::string FeaturesStr;

  // FIXME: We shouldn't need to do this (and link in codegen).
  //        When we split this out, we should do it in a way that makes
  //        it straightforward to switch subtargets on the fly.
  TargetMachine *TM = TheTarget->createTargetMachine(TripleName, FeaturesStr);
  assert(TM && "Unable to create target machine!");

  // Get the target assembler info needed to setup the context.
  const TargetAsmInfo *tai = new TargetAsmInfo(*TM);
  assert(tai && "Unable to create target assembler!");

  // Set up the MCContext for creating symbols and MCExpr's.
  MCContext *Ctx = new MCContext(*MAI, tai);
  assert(Ctx && "Unable to create MCContext!");

  // Set up disassembler.
  MCDisassembler *DisAsm = TheTarget->createMCDisassembler();
  assert(DisAsm && "Unable to create disassembler!");
  DisAsm->setupForSymbolicDisassembly(GetOpInfo, DisInfo, Ctx);

  // Set up the instruction printer.
  int AsmPrinterVariant = MAI->getAssemblerDialect();
  MCInstPrinter *IP = TheTarget->createMCInstPrinter(*TM, AsmPrinterVariant,
                                                     *MAI);
  assert(IP && "Unable to create instruction printer!");

  LLVMDisasmContext *DC = new LLVMDisasmContext(TripleName, DisInfo, TagType,
                                                GetOpInfo, SymbolLookUp,
                                                TheTarget, MAI, TM, tai, Ctx,
                                                DisAsm, IP);
  assert(DC && "Allocation failure!");
  return DC;
}

//
// LLVMDisasmDispose() disposes of the disassembler specified by the context.
//
void LLVMDisasmDispose(LLVMDisasmContextRef DCR){
  LLVMDisasmContext *DC = (LLVMDisasmContext *)DCR;
  delete DC;
}

namespace {
//
// The memory object created by LLVMDisasmInstruction().
//
class DisasmMemoryObject : public MemoryObject {
  uint8_t *Bytes;
  uint64_t Size;
  uint64_t BasePC;
public:
  DisasmMemoryObject(uint8_t *bytes, uint64_t size, uint64_t basePC) :
                     Bytes(bytes), Size(size), BasePC(basePC) {}
 
  uint64_t getBase() const { return BasePC; }
  uint64_t getExtent() const { return Size; }

  int readByte(uint64_t Addr, uint8_t *Byte) const {
    if (Addr - BasePC >= Size)
      return -1;
    *Byte = Bytes[Addr - BasePC];
    return 0;
  }
};
} // end anonymous namespace

//
// LLVMDisasmInstruction() disassembles a single instruction using the
// disassembler context specified in the parameter DC.  The bytes of the
// instruction are specified in the parameter Bytes, and contains at least
// BytesSize number of bytes.  The instruction is at the address specified by
// the PC parameter.  If a valid instruction can be disassembled its string is
// returned indirectly in OutString which whos size is specified in the
// parameter OutStringSize.  This function returns the number of bytes in the
// instruction or zero if there was no valid instruction.  If this function
// returns zero the caller will have to pick how many bytes they want to step
// over by printing a .byte, .long etc. to continue.
//
size_t LLVMDisasmInstruction(LLVMDisasmContextRef DCR, uint8_t *Bytes,
                             uint64_t BytesSize, uint64_t PC, char *OutString,
                             size_t OutStringSize){
  LLVMDisasmContext *DC = (LLVMDisasmContext *)DCR;
  // Wrap the pointer to the Bytes, BytesSize and PC in a MemoryObject.
  DisasmMemoryObject MemoryObject(Bytes, BytesSize, PC);

  uint64_t Size;
  MCInst Inst;
  const MCDisassembler *DisAsm = DC->getDisAsm();
  MCInstPrinter *IP = DC->getIP();
  if (!DisAsm->getInstruction(Inst, Size, MemoryObject, PC, /*REMOVE*/ nulls()))
    return 0;

  SmallVector<char, 64> InsnStr;
  raw_svector_ostream OS(InsnStr);
  IP->printInst(&Inst, OS);
  OS.flush();

  assert(OutStringSize != 0 && "Output buffer cannot be zero size");
  size_t OutputSize = std::min(OutStringSize-1, InsnStr.size());
  std::memcpy(OutString, InsnStr.data(), OutputSize);
  OutString[OutputSize] = '\0'; // Terminate string.

  return Size;
}
