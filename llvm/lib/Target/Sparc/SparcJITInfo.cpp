//===-- SparcJITInfo.cpp - Implement the Sparc JIT Interface --------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the JIT interfaces for the Sparc target.
//
//===----------------------------------------------------------------------===//
#define DEBUG_TYPE "jit"
#include "SparcJITInfo.h"
#include "Sparc.h"
#include "SparcRelocations.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/CodeGen/JITCodeEmitter.h"
#include "llvm/Support/Memory.h"

using namespace llvm;

/// JITCompilerFunction - This contains the address of the JIT function used to
/// compile a function lazily.
static TargetJITInfo::JITCompilerFn JITCompilerFunction;

extern "C" void SparcCompilationCallback();

extern "C" {
#if defined (__sparc__)

#if defined(__arch64__)
#define FRAME_PTR(X) #X "+2047"
#else
#define FRAME_PTR(X) #X
#endif

  asm(
      ".text\n"
      "\t.align 4\n"
      "\t.global SparcCompilationCallback\n"
      "\t.type SparcCompilationCallback, #function\n"
      "SparcCompilationCallback:\n"
      // Save current register window and create stack.
      // 128 (save area) + 6*8 (for arguments) + 16*8 (for float regfile) = 304
      "\tsave %sp, -304, %sp\n"
      // save float regfile to the stack.
      "\tstd %f0,  [" FRAME_PTR(%fp) "-0]\n"
      "\tstd %f2,  [" FRAME_PTR(%fp) "-8]\n"
      "\tstd %f4,  [" FRAME_PTR(%fp) "-16]\n"
      "\tstd %f6,  [" FRAME_PTR(%fp) "-24]\n"
      "\tstd %f8,  [" FRAME_PTR(%fp) "-32]\n"
      "\tstd %f10, [" FRAME_PTR(%fp) "-40]\n"
      "\tstd %f12, [" FRAME_PTR(%fp) "-48]\n"
      "\tstd %f14, [" FRAME_PTR(%fp) "-56]\n"
      "\tstd %f16, [" FRAME_PTR(%fp) "-64]\n"
      "\tstd %f18, [" FRAME_PTR(%fp) "-72]\n"
      "\tstd %f20, [" FRAME_PTR(%fp) "-80]\n"
      "\tstd %f22, [" FRAME_PTR(%fp) "-88]\n"
      "\tstd %f24, [" FRAME_PTR(%fp) "-96]\n"
      "\tstd %f26, [" FRAME_PTR(%fp) "-104]\n"
      "\tstd %f28, [" FRAME_PTR(%fp) "-112]\n"
      "\tstd %f30, [" FRAME_PTR(%fp) "-120]\n"
      // stubaddr is in %g1.
      "\tcall SparcCompilationCallbackC\n"
      "\t  mov %g1, %o0\n"
      // restore float regfile from the stack.
      "\tldd [" FRAME_PTR(%fp) "-0],   %f0\n"
      "\tldd [" FRAME_PTR(%fp) "-8],   %f2\n"
      "\tldd [" FRAME_PTR(%fp) "-16],  %f4\n"
      "\tldd [" FRAME_PTR(%fp) "-24],  %f6\n"
      "\tldd [" FRAME_PTR(%fp) "-32],  %f8\n"
      "\tldd [" FRAME_PTR(%fp) "-40],  %f10\n"
      "\tldd [" FRAME_PTR(%fp) "-48],  %f12\n"
      "\tldd [" FRAME_PTR(%fp) "-56],  %f14\n"
      "\tldd [" FRAME_PTR(%fp) "-64],  %f16\n"
      "\tldd [" FRAME_PTR(%fp) "-72],  %f18\n"
      "\tldd [" FRAME_PTR(%fp) "-80],  %f20\n"
      "\tldd [" FRAME_PTR(%fp) "-88],  %f22\n"
      "\tldd [" FRAME_PTR(%fp) "-96],  %f24\n"
      "\tldd [" FRAME_PTR(%fp) "-104], %f26\n"
      "\tldd [" FRAME_PTR(%fp) "-112], %f28\n"
      "\tldd [" FRAME_PTR(%fp) "-120], %f30\n"
      // restore original register window and
      // copy %o0 to %g1
      "\trestore %o0, 0, %g1\n"
      // call the new stub
      "\tjmp %g1\n"
      "\t  nop\n"
      "\t.size   SparcCompilationCallback, .-SparcCompilationCallback"
      );
#else
  void SparcCompilationCallback() {
    llvm_unreachable(
      "Cannot call SparcCompilationCallback() on a non-sparc arch!");
  }
#endif
}


#define SETHI_INST(imm, rd)    (0x01000000 | ((rd) << 25) | ((imm) & 0x3FFFFF))
#define JMP_INST(rs1, imm, rd) (0x80000000 | ((rd) << 25) | (0x38 << 19) \
                                | ((rs1) << 14) | (1 << 13) | ((imm) & 0x1FFF))
#define NOP_INST               SETHI_INST(0, 0)
#define OR_INST_I(rs1, imm, rd) (0x80000000 | ((rd) << 25) | (0x02 << 19) \
                                 | ((rs1) << 14) | (1 << 13) | ((imm) & 0x1FFF))
#define OR_INST_R(rs1, rs2, rd) (0x80000000 | ((rd) << 25) | (0x02 << 19) \
                                 | ((rs1) << 14) | (0 << 13) | ((rs2) & 0x1F))
#define RDPC_INST(rd)           (0x80000000 | ((rd) << 25) | (0x28 << 19) \
                                 | (5 << 14))
#define LDX_INST(rs1, imm, rd)  (0xC0000000 | ((rd) << 25) | (0x0B << 19) \
                                 | ((rs1) << 14) | (1 << 13) | ((imm) & 0x1FFF))
#define SLLX_INST(rs1, imm, rd) (0x80000000 | ((rd) << 25) | (0x25 << 19) \
                                 | ((rs1) << 14) | (3 << 12) | ((imm) & 0x3F))
#define SUB_INST(rs1, imm, rd)  (0x80000000 | ((rd) << 25) | (0x04 << 19) \
                                 | ((rs1) << 14) | (1 << 13) | ((imm) & 0x1FFF))
#define XOR_INST(rs1, imm, rd)  (0x80000000 | ((rd) << 25) | (0x03 << 19) \
                                 | ((rs1) << 14) | (1 << 13) | ((imm) & 0x1FFF))
#define BA_INST(tgt)             (0x10800000 | ((tgt) & 0x3FFFFF))

// Emit instructions to jump to Addr and store the starting address of
// the instructions emitted in the scratch register.
static void emitInstrForIndirectJump(intptr_t Addr,
                                     unsigned scratch,
                                     SmallVectorImpl<uint32_t> &Insts) {

  if (isInt<13>(Addr)) {
    // Emit: jmpl %g0+Addr, <scratch>
    //         nop
    Insts.push_back(JMP_INST(0, LO10(Addr), scratch));
    Insts.push_back(NOP_INST);
    return;
  }

  if (isUInt<32>(Addr)) {
    // Emit: sethi %hi(Addr), scratch
    //       jmpl scratch+%lo(Addr), scratch
    //         sub scratch, 4, scratch
    Insts.push_back(SETHI_INST(HI22(Addr), scratch));
    Insts.push_back(JMP_INST(scratch, LO10(Addr), scratch));
    Insts.push_back(SUB_INST(scratch, 4, scratch));
    return;
  }

  if (Addr < 0 && isInt<33>(Addr)) {
    // Emit: sethi %hix(Addr), scratch)
    //       xor   scratch, %lox(Addr), scratch
    //       jmpl scratch+0, scratch
    //         sub scratch, 8, scratch
    Insts.push_back(SETHI_INST(HIX22(Addr), scratch));
    Insts.push_back(XOR_INST(scratch, LOX10(Addr), scratch));
    Insts.push_back(JMP_INST(scratch, 0, scratch));
    Insts.push_back(SUB_INST(scratch, 8, scratch));
    return;
  }

  // Emit: rd %pc, scratch
  //       ldx [scratch+16], scratch
  //       jmpl scratch+0, scratch
  //         sub scratch, 8, scratch
  //       <Addr: 8 byte>
  Insts.push_back(RDPC_INST(scratch));
  Insts.push_back(LDX_INST(scratch, 16, scratch));
  Insts.push_back(JMP_INST(scratch, 0, scratch));
  Insts.push_back(SUB_INST(scratch, 8, scratch));
  Insts.push_back((uint32_t)(((int64_t)Addr) >> 32) & 0xffffffff);
  Insts.push_back((uint32_t)(Addr & 0xffffffff));

  // Instruction sequence without rdpc instruction
  // 7 instruction and 2 scratch register
  // Emit: sethi %hh(Addr), scratch
  //       or scratch, %hm(Addr), scratch
  //       sllx scratch, 32, scratch
  //       sethi %hi(Addr), scratch2
  //       or scratch, scratch2, scratch
  //       jmpl scratch+%lo(Addr), scratch
  //         sub scratch, 20, scratch
  // Insts.push_back(SETHI_INST(HH22(Addr), scratch));
  // Insts.push_back(OR_INST_I(scratch, HM10(Addr), scratch));
  // Insts.push_back(SLLX_INST(scratch, 32, scratch));
  // Insts.push_back(SETHI_INST(HI22(Addr), scratch2));
  // Insts.push_back(OR_INST_R(scratch, scratch2, scratch));
  // Insts.push_back(JMP_INST(scratch, LO10(Addr), scratch));
  // Insts.push_back(SUB_INST(scratch, 20, scratch));
}

extern "C" void *SparcCompilationCallbackC(intptr_t StubAddr) {
  // Get the address of the compiled code for this function.
  intptr_t NewVal = (intptr_t) JITCompilerFunction((void*) StubAddr);

  // Rewrite the function stub so that we don't end up here every time we
  // execute the call. We're replacing the stub instructions with code
  // that jumps to the compiled function:

  SmallVector<uint32_t, 8> Insts;
  intptr_t diff = (NewVal - StubAddr) >> 2;
  if (isInt<22>(diff)) {
    // Use branch instruction to jump
    Insts.push_back(BA_INST(diff));
    Insts.push_back(NOP_INST);
  } else {
    // Otherwise, use indirect jump to the compiled function
    emitInstrForIndirectJump(NewVal, 1, Insts);
  }

  for (unsigned i = 0, e = Insts.size(); i != e; ++i)
    *(uint32_t *)(StubAddr + i*4) = Insts[i];

  sys::Memory::InvalidateInstructionCache((void*) StubAddr, Insts.size() * 4);
  return (void*)StubAddr;
}


void SparcJITInfo::replaceMachineCodeForFunction(void *Old, void *New) {
  assert(0 && "FIXME: Implement SparcJITInfo::replaceMachineCodeForFunction");
}


TargetJITInfo::StubLayout SparcJITInfo::getStubLayout() {
  // The stub contains maximum of 4 4-byte instructions and 8 bytes for address,
  // aligned at 32 bytes.
  // See emitFunctionStub and emitInstrForIndirectJump for details.
  StubLayout Result = { 4*4 + 8, 32 };
  return Result;
}

void *SparcJITInfo::emitFunctionStub(const Function *F, void *Fn,
                                     JITCodeEmitter &JCE)
{
  JCE.emitAlignment(32);
  void *Addr = (void*) (JCE.getCurrentPCValue());

  intptr_t CurrentAddr = (intptr_t)Addr;
  intptr_t EmittedAddr;
  SmallVector<uint32_t, 8> Insts;
  if (Fn != (void*)(intptr_t)SparcCompilationCallback) {
    EmittedAddr = (intptr_t)Fn;
    intptr_t diff = (EmittedAddr - CurrentAddr) >> 2;
    if (isInt<22>(diff)) {
      Insts.push_back(BA_INST(diff));
      Insts.push_back(NOP_INST);
    }
  } else {
    EmittedAddr = (intptr_t)SparcCompilationCallback;
  }

  if (Insts.size() == 0)
    emitInstrForIndirectJump(EmittedAddr, 1, Insts);


  if (!sys::Memory::setRangeWritable(Addr, 4 * Insts.size()))
    llvm_unreachable("ERROR: Unable to mark stub writable.");

  for (unsigned i = 0, e = Insts.size(); i != e; ++i)
    JCE.emitWordBE(Insts[i]);

  sys::Memory::InvalidateInstructionCache(Addr, 4 * Insts.size());
  if (!sys::Memory::setRangeExecutable(Addr, 4 * Insts.size()))
    llvm_unreachable("ERROR: Unable to mark stub executable.");

  return Addr;
}


TargetJITInfo::LazyResolverFn
SparcJITInfo::getLazyResolverFunction(JITCompilerFn F) {
  JITCompilerFunction = F;
  return SparcCompilationCallback;
}

/// relocate - Before the JIT can run a block of code that has been emitted,
/// it must rewrite the code to contain the actual addresses of any
/// referenced global symbols.
void SparcJITInfo::relocate(void *Function, MachineRelocation *MR,
                            unsigned NumRelocs, unsigned char *GOTBase) {
  for (unsigned i = 0; i != NumRelocs; ++i, ++MR) {
    void *RelocPos = (char*) Function + MR->getMachineCodeOffset();
    intptr_t ResultPtr = (intptr_t) MR->getResultPointer();

    switch ((SP::RelocationType) MR->getRelocationType()) {
    case SP::reloc_sparc_hi:
      ResultPtr = (ResultPtr >> 10) & 0x3fffff;
      break;

    case SP::reloc_sparc_lo:
      ResultPtr = (ResultPtr & 0x3ff);
      break;

    case SP::reloc_sparc_pc30:
      ResultPtr = ((ResultPtr - (intptr_t)RelocPos) >> 2) & 0x3fffffff;
      break;

    case SP::reloc_sparc_pc22:
      ResultPtr = ((ResultPtr - (intptr_t)RelocPos) >> 2) & 0x3fffff;
      break;

    case SP::reloc_sparc_pc19:
      ResultPtr = ((ResultPtr - (intptr_t)RelocPos) >> 2) & 0x7ffff;
      break;

    case SP::reloc_sparc_h44:
      ResultPtr = (ResultPtr >> 22) & 0x3fffff;
      break;

    case SP::reloc_sparc_m44:
      ResultPtr = (ResultPtr >> 12) & 0x3ff;
      break;

    case SP::reloc_sparc_l44:
      ResultPtr = (ResultPtr & 0xfff);
      break;

    case SP::reloc_sparc_hh:
      ResultPtr = (((int64_t)ResultPtr) >> 42) & 0x3fffff;
      break;

    case SP::reloc_sparc_hm:
      ResultPtr = (((int64_t)ResultPtr) >> 32) & 0x3ff;
      break;

    }
    *((unsigned*) RelocPos) |= (unsigned) ResultPtr;
  }
}
