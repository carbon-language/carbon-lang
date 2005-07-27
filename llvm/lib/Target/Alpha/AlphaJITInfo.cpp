//===-- AlphaJITInfo.cpp - Implement the JIT interfaces for the Alpha ---===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the JIT interfaces for the Alpha target.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "jit"
#include "AlphaJITInfo.h"
#include "AlphaRelocations.h"
#include "llvm/CodeGen/MachineCodeEmitter.h"
#include "llvm/Config/alloca.h"
#include "llvm/Support/Debug.h"
#include <cstdlib>
#include <iostream>
#include <map>
using namespace std;
using namespace llvm;

#define BUILD_LDA(RD, RS, IMM16) \
  ((0x08 << 26) | ((RD) << 21) | ((RS) << 16) | ((IMM16) & 65535))
#define BUILD_LDAH(RD, RS, IMM16) \
  ((0x09 << 26) | ((RD) << 21) | ((RS) << 16) | ((IMM16) & 65535))

#define MERGE_PARTS(HH, HL, LH, LL) \
  (((HH * 65536 + HL) << 32) + (LH * 65536 + LL))

#define BUILD_LDQ(RD, RS, IMM16) \
  ((0x29 << 26) | ((RD) << 21) | ((RS) << 16) | ((IMM16) & 0xFFFF))

#define BUILD_JMP(RD, RS, IMM16) \
  ((0x1A << 26) | ((RD) << 21) | ((RS) << 16) | ((IMM16) & 0xFFFF))

static void EmitBranchToAt(void *At, void *To, bool isCall) {
  //FIXME
  assert(0);
}

void AlphaJITInfo::replaceMachineCodeForFunction(void *Old, void *New) {
  //FIXME
  assert(0);
}

static TargetJITInfo::JITCompilerFn JITCompilerFunction;
//static AlphaJITInfo* AlphaJTI;

extern "C" {
#ifdef __alpha

  void AlphaCompilationCallbackC(long* oldsp)
  {
    void* CameFromStub = (void*)*(oldsp - 1);
    void* CameFromOrig = (void*)*(oldsp - 2);

    void* Target = JITCompilerFunction(CameFromStub);

    //rewrite the stub to an unconditional branch
    EmitBranchToAt(CameFromStub, Target, false);

    //Change pv to new Target
    *(oldsp - 1) = (long)Target;

    //special epilog
    register long* RSP asm ("$0") = oldsp;
    __asm__ __volatile__ (
      "ldq $16,   0($0)\n"
      "ldq $17,   8($0)\n"
      "ldq $18,  16($0)\n"
      "ldq $19,  24($0)\n"
      "ldq $20,  32($0)\n"
      "ldq $21,  40($0)\n"
      "ldt $f16, 48($0)\n"
      "ldt $f17, 56($0)\n"
      "ldt $f18, 64($0)\n"
      "ldt $f19, 72($0)\n"
      "ldt $f20, 80($0)\n"
      "ldt $f21, 88($0)\n"
      "ldq $9,   96($0)\n"
      "ldq $10, 104($0)\n"
      "ldq $11, 112($0)\n"
      "ldq $12, 120($0)\n"
      "ldq $13, 128($0)\n"
      "ldq $14, 136($0)\n"
      "ldt $f2, 144($0)\n"
      "ldt $f3, 152($0)\n"
      "ldt $f4, 160($0)\n"
      "ldt $f5, 168($0)\n"
      "ldt $f6, 176($0)\n"
      "ldt $f7, 184($0)\n"
      "ldt $f8, 192($0)\n"
      "ldt $f9, 200($0)\n"
      "ldq $15, 208($0)\n"
      "ldq $26, 216($0)\n"
      "ldq $27, 224($0)\n"
      "bis $30, $0, $0\n" //restore sp
      "jmp $31, ($27)\n" //jump to the new function
      "and $0, $31, $31\n" //dummy use of r0
    );
  }

  void AlphaCompilationCallback(void);

  asm(
      ".text\n"
      ".globl AlphaComilationCallbackC\n"
      ".align 4\n"
      ".globl AlphaCompilationCallback\n"
      ".ent AlphaCompilationCallback\n"
"AlphaCompilationCallback:\n"
      //      //get JIT's GOT
      //      "ldgp\n"
      //Save args, callee saved, and perhaps others?
      //args: $16-$21 $f16-$f21     (12)
      //callee: $9-$14 $f2-$f9      (14)
      //others: fp:$15 ra:$26 pv:$27 (3)
      "bis $0, $30, $30\n" //0 = sp
      "lda $30, -232($30)\n"
      "stq $16,   0($30)\n"
      "stq $17,   8($30)\n"
      "stq $18,  16($30)\n"
      "stq $19,  24($30)\n"
      "stq $20,  32($30)\n"
      "stq $21,  40($30)\n"
      "stt $f16, 48($30)\n"
      "stt $f17, 56($30)\n"
      "stt $f18, 64($30)\n"
      "stt $f19, 72($30)\n"
      "stt $f20, 80($30)\n"
      "stt $f21, 88($30)\n"
      "stq $9,   96($30)\n"
      "stq $10, 104($30)\n"
      "stq $11, 112($30)\n"
      "stq $12, 120($30)\n"
      "stq $13, 128($30)\n"
      "stq $14, 136($30)\n"
      "stt $f2, 144($30)\n"
      "stt $f3, 152($30)\n"
      "stt $f4, 160($30)\n"
      "stt $f5, 168($30)\n"
      "stt $f6, 176($30)\n"
      "stt $f7, 184($30)\n"
      "stt $f8, 192($30)\n"
      "stt $f9, 200($30)\n"
      "stq $15, 208($30)\n"
      "stq $26, 216($30)\n"
      "stq $27, 224($30)\n"
      "bis $16, $0, $0\n" //pass the old sp as the first arg
      "bsr $31, AlphaCompilationCallbackC\n"
      ".end AlphaCompilationCallback\n"
      );
#else
  void AlphaCompilationCallback() {
    std::cerr << "Cannot call AlphaCompilationCallback() on a non-Alpha arch!\n";
    abort();
  }
#endif
}

void *AlphaJITInfo::emitFunctionStub(void *Fn, MachineCodeEmitter &MCE) {
//   // If this is just a call to an external function, emit a branch instead of a
//   // call.  This means looking up Fn and storing that in R27 so as to appear to
//   // have called there originally
//   if (Fn != AlphaCompilationCallback) {
//     int idx = AlphaJTI->getNewGOTEntry(Fn);
//     //R27 = ldq idx(R29)
//     //R31 = JMP R27, 0
//     MCE.startFunctionStub(2*4);
//     void *Addr = (void*)(intptr_t)MCE.getCurrentPCValue();
//     MCE.emitWord(BUILD_LDQ(27, 29, idx << 3));
//     MCE.emitWord(BUILD_JMP(31, 27, 0));
//     return MCE.finishFunctionStub(0);
//   }

  assert(0 && "Need to be able to jump to this guy too");
}

TargetJITInfo::LazyResolverFn
AlphaJITInfo::getLazyResolverFunction(JITCompilerFn F) {
  JITCompilerFunction = F;
  //  setZerothGOTEntry((void*)AlphaCompilationCallback);
  return AlphaCompilationCallback;
}

//These describe LDAx
static const int IMM_LOW  = -32768;
static const int IMM_HIGH = 32767;
static const int IMM_MULT = 65536;

static long getUpper16(long l)
{
  long y = l / IMM_MULT;
  if (l % IMM_MULT > IMM_HIGH)
    ++y;
  if (l % IMM_MULT < IMM_LOW)
    --y;
  assert((short)y == y && "displacement out of range");
  return y;
}

static long getLower16(long l)
{
  long h = getUpper16(l);
  long y = l - h * IMM_MULT;
  assert(y == (short)y && "Displacement out of range");
  return y;
}

void AlphaJITInfo::relocate(void *Function, MachineRelocation *MR,
                            unsigned NumRelocs, unsigned char* GOTBase) {
  //because gpdist are paired and relative to the pc of the first inst,
  //we need to have some state

  static map<pair<void*, int>, void*> gpdistmap;

  for (unsigned i = 0; i != NumRelocs; ++i, ++MR) {
    unsigned *RelocPos = (unsigned*)Function + MR->getMachineCodeOffset()/4;
    long idx = 0;
    switch ((Alpha::RelocationType)MR->getRelocationType()) {
    default: assert(0 && "Unknown relocation type!");
    case Alpha::reloc_literal:
      //This is a LDQl
      idx = MR->getGOTIndex();
      DEBUG(std::cerr << "Literal relocation to slot " << idx);
      idx = (idx - GOToffset) * 8;
      DEBUG(std::cerr << " offset " << idx << "\n");
      break;
    case Alpha::reloc_gprellow:
      idx = (unsigned char*)MR->getResultPointer() - &GOTBase[GOToffset * 8];
      idx = getLower16(idx);
      DEBUG(std::cerr << "gprellow relocation offset " << idx << "\n");
      DEBUG(std::cerr << " Pointer is " << (void*)MR->getResultPointer()
            << " GOT is " << (void*)&GOTBase[GOToffset * 8] << "\n");
      break;
    case Alpha::reloc_gprelhigh:
      idx = (unsigned char*)MR->getResultPointer() - &GOTBase[GOToffset * 8];
      idx = getUpper16(idx);
      DEBUG(std::cerr << "gprelhigh relocation offset " << idx << "\n");
      DEBUG(std::cerr << " Pointer is " << (void*)MR->getResultPointer()
            << " GOT is " << (void*)&GOTBase[GOToffset * 8] << "\n");
      break;
    case Alpha::reloc_gpdist:
      switch (*RelocPos >> 26) {
      case 0x09: //LDAH
        idx = &GOTBase[GOToffset * 8] - (unsigned char*)RelocPos;
        idx = getUpper16(idx);
        DEBUG(std::cerr << "LDAH: " << idx << "\n");
        //add the relocation to the map
        gpdistmap[make_pair(Function, MR->getConstantVal())] = RelocPos;
        break;
      case 0x08: //LDA
        assert(gpdistmap[make_pair(Function, MR->getConstantVal())] &&
               "LDAg without seeing LDAHg");
        idx = &GOTBase[GOToffset * 8] -
          (unsigned char*)gpdistmap[make_pair(Function, MR->getConstantVal())];
        idx = getLower16(idx);
        DEBUG(std::cerr << "LDA: " << idx << "\n");
        break;
      default:
        assert(0 && "Cannot handle gpdist yet");
      }
      break;
    }
    short x = (short)idx;
    assert(x == idx);
    *(short*)RelocPos = x;
  }
}
