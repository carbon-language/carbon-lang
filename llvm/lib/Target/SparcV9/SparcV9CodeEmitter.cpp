//===-- SparcV9CodeEmitter.cpp --------------------------------------------===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// SPARC-specific backend for emitting machine code to memory.
//
// This module also contains the code for lazily resolving the targets
// of call instructions, including the callback used to redirect calls
// to functions for which the code has not yet been generated into the
// JIT compiler.
//
// This file #includes SparcV9CodeEmitter.inc, which contains the code
// for getBinaryCodeForInstr(), a method that converts a MachineInstr
// into the corresponding binary machine code word.
//
//===----------------------------------------------------------------------===//

#include "llvm/Constants.h"
#include "llvm/Function.h"
#include "llvm/GlobalVariable.h"
#include "llvm/PassManager.h"
#include "llvm/CodeGen/MachineCodeEmitter.h"
#include "llvm/CodeGen/MachineConstantPool.h"
#include "llvm/CodeGen/MachineFunctionInfo.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetData.h"
#include "Support/Debug.h"
#include "Support/hash_set"
#include "Support/Statistic.h"
#include "SparcV9Internals.h"
#include "SparcV9TargetMachine.h"
#include "SparcV9RegInfo.h"
#include "SparcV9CodeEmitter.h"
#include "Config/alloca.h"

namespace llvm {

namespace {
  Statistic<> OverwrittenCalls("call-ovwr", "Number of over-written calls");
  Statistic<> UnmodifiedCalls("call-skip", "Number of unmodified calls");
  Statistic<> CallbackCalls("callback", "Number CompilationCallback() calls");
}

bool SparcV9TargetMachine::addPassesToEmitMachineCode(FunctionPassManager &PM,
                                                    MachineCodeEmitter &MCE) {
  MachineCodeEmitter *M = &MCE;
  DEBUG(M = MachineCodeEmitter::createFilePrinterEmitter(MCE));
  PM.add(new SparcV9CodeEmitter(*this, *M));
  PM.add(createSparcV9MachineCodeDestructionPass()); //Free stuff no longer needed
  return false;
}

namespace {
  class JITResolver {
    SparcV9CodeEmitter &SparcV9;
    MachineCodeEmitter &MCE;

    /// LazyCodeGenMap - Keep track of call sites for functions that are to be
    /// lazily resolved.
    ///
    std::map<uint64_t, Function*> LazyCodeGenMap;

    /// LazyResolverMap - Keep track of the lazy resolver created for a
    /// particular function so that we can reuse them if necessary.
    ///
    std::map<Function*, uint64_t> LazyResolverMap;

  public:
    enum CallType { ShortCall, FarCall };

  private:
    /// We need to keep track of whether we used a simple call or a far call
    /// (many instructions) in sequence. This means we need to keep track of
    /// what type of stub we generate.
    static std::map<uint64_t, CallType> LazyCallFlavor;

  public:
    JITResolver(SparcV9CodeEmitter &V9,
                MachineCodeEmitter &mce) : SparcV9(V9), MCE(mce) {}
    uint64_t getLazyResolver(Function *F);
    uint64_t addFunctionReference(uint64_t Address, Function *F);
    void deleteFunctionReference(uint64_t Address);
    void addCallFlavor(uint64_t Address, CallType Flavor) {
      LazyCallFlavor[Address] = Flavor;
    }

    // Utility functions for accessing data from static callback
    uint64_t getCurrentPCValue() {
      return MCE.getCurrentPCValue();
    }
    unsigned getBinaryCodeForInstr(MachineInstr &MI) {
      return SparcV9.getBinaryCodeForInstr(MI);
    }

    inline void insertFarJumpAtAddr(int64_t Value, uint64_t Addr);
    void insertJumpAtAddr(int64_t Value, uint64_t &Addr);

  private:
    uint64_t emitStubForFunction(Function *F);
    static void SaveRegisters(uint64_t DoubleFP[], uint64_t CC[],
                              uint64_t Globals[]);
    static void RestoreRegisters(uint64_t DoubleFP[], uint64_t CC[],
                                 uint64_t Globals[]);
    static void CompilationCallback();
    uint64_t resolveFunctionReference(uint64_t RetAddr);

  };

  JITResolver *TheJITResolver;
  std::map<uint64_t, JITResolver::CallType> JITResolver::LazyCallFlavor;
}

/// addFunctionReference - This method is called when we need to emit the
/// address of a function that has not yet been emitted, so we don't know the
/// address.  Instead, we emit a call to the CompilationCallback method, and
/// keep track of where we are.
///
uint64_t JITResolver::addFunctionReference(uint64_t Address, Function *F) {
  LazyCodeGenMap[Address] = F;
  return (intptr_t)&JITResolver::CompilationCallback;
}

/// deleteFunctionReference - If we are emitting a far call, we already added a
/// reference to the function, but it is now incorrect, since the address to the
/// JIT resolver is too far away to be a simple call instruction. This is used
/// to remove the address from the map.
///
void JITResolver::deleteFunctionReference(uint64_t Address) {
  std::map<uint64_t, Function*>::iterator I = LazyCodeGenMap.find(Address);
  assert(I != LazyCodeGenMap.end() && "Not in map!");
  LazyCodeGenMap.erase(I);  
}

uint64_t JITResolver::resolveFunctionReference(uint64_t RetAddr) {
  std::map<uint64_t, Function*>::iterator I = LazyCodeGenMap.find(RetAddr);
  assert(I != LazyCodeGenMap.end() && "Not in map!");
  Function *F = I->second;
  LazyCodeGenMap.erase(I);
  return MCE.forceCompilationOf(F);
}

uint64_t JITResolver::getLazyResolver(Function *F) {
  std::map<Function*, uint64_t>::iterator I = LazyResolverMap.lower_bound(F);
  if (I != LazyResolverMap.end() && I->first == F) return I->second;
  
  uint64_t Stub = emitStubForFunction(F);
  LazyResolverMap.insert(I, std::make_pair(F, Stub));
  return Stub;
}

void JITResolver::insertJumpAtAddr(int64_t JumpTarget, uint64_t &Addr) {
  DEBUG(std::cerr << "Emitting a jump to 0x" << std::hex << JumpTarget << "\n");

  // If the target function is close enough to fit into the 19bit disp of
  // BA, we should use this version, as it's much cheaper to generate.
  int64_t BranchTarget = (JumpTarget-Addr) >> 2;
  if (BranchTarget >= (1 << 19) || BranchTarget <= -(1 << 19)) {
    TheJITResolver->insertFarJumpAtAddr(JumpTarget, Addr);
  } else {
    // ba <target>
    MachineInstr *I = BuildMI(V9::BA, 1).addSImm(BranchTarget);
    *((unsigned*)(intptr_t)Addr) = getBinaryCodeForInstr(*I);
    Addr += 4;
    delete I;

    // nop
    I = BuildMI(V9::NOP, 0);
    *((unsigned*)(intptr_t)Addr) = getBinaryCodeForInstr(*I);
    delete I;
  }
}

void JITResolver::insertFarJumpAtAddr(int64_t Target, uint64_t Addr) {
  static const unsigned 
    o6 = SparcV9IntRegClass::o6, g0 = SparcV9IntRegClass::g0,
    g1 = SparcV9IntRegClass::g1, g5 = SparcV9IntRegClass::g5;

  MachineInstr* BinaryCode[] = {
    //
    // Get address to branch into %g1, using %g5 as a temporary
    //
    // sethi %uhi(Target), %g5     ;; get upper 22 bits of Target into %g5
    BuildMI(V9::SETHI, 2).addSImm(Target >> 42).addReg(g5),
    // or %g5, %ulo(Target), %g5   ;; get 10 lower bits of upper word into %g5
    BuildMI(V9::ORi, 3).addReg(g5).addSImm((Target >> 32) & 0x03ff).addReg(g5),
    // sllx %g5, 32, %g5           ;; shift those 10 bits to the upper word
    BuildMI(V9::SLLXi6, 3).addReg(g5).addSImm(32).addReg(g5),
    // sethi %hi(Target), %g1      ;; extract bits 10-31 into the dest reg
    BuildMI(V9::SETHI, 2).addSImm((Target >> 10) & 0x03fffff).addReg(g1),
    // or %g5, %g1, %g1            ;; get upper word (in %g5) into %g1
    BuildMI(V9::ORr, 3).addReg(g5).addReg(g1).addReg(g1),
    // or %g1, %lo(Target), %g1    ;; get lowest 10 bits of Target into %g1
    BuildMI(V9::ORi, 3).addReg(g1).addSImm(Target & 0x03ff).addReg(g1),
    // jmpl %g1, %g0, %g0          ;; indirect branch on %g1
    BuildMI(V9::JMPLRETr, 3).addReg(g1).addReg(g0).addReg(g0),
    // nop                         ;; delay slot
    BuildMI(V9::NOP, 0)
  };

  for (unsigned i=0, e=sizeof(BinaryCode)/sizeof(BinaryCode[0]); i!=e; ++i) {
    *((unsigned*)(intptr_t)Addr) = getBinaryCodeForInstr(*BinaryCode[i]);
    delete BinaryCode[i];
    Addr += 4;
  }
}

void JITResolver::SaveRegisters(uint64_t DoubleFP[], uint64_t CC[], 
                                uint64_t Globals[]) {
#if defined(sparc) || defined(__sparc__) || defined(__sparcv9)

  __asm__ __volatile__ (// Save condition-code registers
                        "stx %%fsr, %0;\n\t" 
                        "rd %%fprs, %1;\n\t" 
                        "rd %%ccr,  %2;\n\t"
                        : "=m"(CC[0]), "=r"(CC[1]), "=r"(CC[2]));

  __asm__ __volatile__ (// Save globals g1 and g5
                        "stx %%g1, %0;\n\t"
                        "stx %%g5, %0;\n\t"
                        : "=m"(Globals[0]), "=m"(Globals[1]));

  // GCC says: `asm' only allows up to thirty parameters!
  __asm__ __volatile__ (// Save Single/Double FP registers, part 1
                        "std  %%f0,  %0;\n\t"  "std  %%f2,  %1;\n\t"
                        "std  %%f4,  %2;\n\t"  "std  %%f6,  %3;\n\t"
                        "std  %%f8,  %4;\n\t"  "std  %%f10, %5;\n\t"
                        "std  %%f12, %6;\n\t"  "std  %%f14, %7;\n\t"
                        "std  %%f16, %8;\n\t"  "std  %%f18, %9;\n\t"
                        "std  %%f20, %10;\n\t" "std  %%f22, %11;\n\t"
                        "std  %%f24, %12;\n\t" "std  %%f26, %13;\n\t"
                        "std  %%f28, %14;\n\t" "std  %%f30, %15;\n\t"
                        : "=m"(DoubleFP[ 0]), "=m"(DoubleFP[ 1]),
                          "=m"(DoubleFP[ 2]), "=m"(DoubleFP[ 3]),
                          "=m"(DoubleFP[ 4]), "=m"(DoubleFP[ 5]),
                          "=m"(DoubleFP[ 6]), "=m"(DoubleFP[ 7]),
                          "=m"(DoubleFP[ 8]), "=m"(DoubleFP[ 9]),
                          "=m"(DoubleFP[10]), "=m"(DoubleFP[11]),
                          "=m"(DoubleFP[12]), "=m"(DoubleFP[13]),
                          "=m"(DoubleFP[14]), "=m"(DoubleFP[15]));
                        
  __asm__ __volatile__ (// Save Double FP registers, part 2
                        "std %%f32, %0;\n\t"  "std %%f34, %1;\n\t"
                        "std %%f36, %2;\n\t"  "std %%f38, %3;\n\t"
                        "std %%f40, %4;\n\t"  "std %%f42, %5;\n\t"
                        "std %%f44, %6;\n\t"  "std %%f46, %7;\n\t"
                        "std %%f48, %8;\n\t"  "std %%f50, %9;\n\t"
                        "std %%f52, %10;\n\t" "std %%f54, %11;\n\t"
                        "std %%f56, %12;\n\t" "std %%f58, %13;\n\t"
                        "std %%f60, %14;\n\t" "std %%f62, %15;\n\t"
                        : "=m"(DoubleFP[16]), "=m"(DoubleFP[17]),
                          "=m"(DoubleFP[18]), "=m"(DoubleFP[19]),
                          "=m"(DoubleFP[20]), "=m"(DoubleFP[21]),
                          "=m"(DoubleFP[22]), "=m"(DoubleFP[23]),
                          "=m"(DoubleFP[24]), "=m"(DoubleFP[25]),
                          "=m"(DoubleFP[26]), "=m"(DoubleFP[27]),
                          "=m"(DoubleFP[28]), "=m"(DoubleFP[29]),
                          "=m"(DoubleFP[30]), "=m"(DoubleFP[31]));
#endif
}


void JITResolver::RestoreRegisters(uint64_t DoubleFP[], uint64_t CC[], 
                                   uint64_t Globals[])
{
#if defined(sparc) || defined(__sparc__) || defined(__sparcv9)

  __asm__ __volatile__ (// Restore condition-code registers
                        "ldx %0,    %%fsr;\n\t" 
                        "wr  %1, 0, %%fprs;\n\t"
                        "wr  %2, 0, %%ccr;\n\t" 
                        :: "m"(CC[0]), "r"(CC[1]), "r"(CC[2]));

  __asm__ __volatile__ (// Restore globals g1 and g5
                        "ldx %0, %%g1;\n\t"
                        "ldx %0, %%g5;\n\t"
                        :: "m"(Globals[0]), "m"(Globals[1]));

  // GCC says: `asm' only allows up to thirty parameters!
  __asm__ __volatile__ (// Restore Single/Double FP registers, part 1
                        "ldd %0,  %%f0;\n\t"   "ldd %1, %%f2;\n\t" 
                        "ldd %2,  %%f4;\n\t"   "ldd %3, %%f6;\n\t" 
                        "ldd %4,  %%f8;\n\t"   "ldd %5, %%f10;\n\t" 
                        "ldd %6,  %%f12;\n\t"  "ldd %7, %%f14;\n\t" 
                        "ldd %8,  %%f16;\n\t"  "ldd %9, %%f18;\n\t" 
                        "ldd %10, %%f20;\n\t" "ldd %11, %%f22;\n\t"
                        "ldd %12, %%f24;\n\t" "ldd %13, %%f26;\n\t"
                        "ldd %14, %%f28;\n\t" "ldd %15, %%f30;\n\t"
                        :: "m"(DoubleFP[0]), "m"(DoubleFP[1]),
                           "m"(DoubleFP[2]), "m"(DoubleFP[3]),
                           "m"(DoubleFP[4]), "m"(DoubleFP[5]),
                           "m"(DoubleFP[6]), "m"(DoubleFP[7]),
                           "m"(DoubleFP[8]), "m"(DoubleFP[9]),
                           "m"(DoubleFP[10]), "m"(DoubleFP[11]),
                           "m"(DoubleFP[12]), "m"(DoubleFP[13]),
                           "m"(DoubleFP[14]), "m"(DoubleFP[15]));

  __asm__ __volatile__ (// Restore Double FP registers, part 2
                        "ldd %0, %%f32;\n\t"  "ldd %1, %%f34;\n\t"
                        "ldd %2, %%f36;\n\t"  "ldd %3, %%f38;\n\t"
                        "ldd %4, %%f40;\n\t"  "ldd %5, %%f42;\n\t"
                        "ldd %6, %%f44;\n\t"  "ldd %7, %%f46;\n\t"
                        "ldd %8, %%f48;\n\t"  "ldd %9, %%f50;\n\t"
                        "ldd %10, %%f52;\n\t" "ldd %11, %%f54;\n\t"
                        "ldd %12, %%f56;\n\t" "ldd %13, %%f58;\n\t"
                        "ldd %14, %%f60;\n\t" "ldd %15, %%f62;\n\t"
                        :: "m"(DoubleFP[16]), "m"(DoubleFP[17]),
                           "m"(DoubleFP[18]), "m"(DoubleFP[19]),
                           "m"(DoubleFP[20]), "m"(DoubleFP[21]),
                           "m"(DoubleFP[22]), "m"(DoubleFP[23]),
                           "m"(DoubleFP[24]), "m"(DoubleFP[25]),
                           "m"(DoubleFP[26]), "m"(DoubleFP[27]),
                           "m"(DoubleFP[28]), "m"(DoubleFP[29]),
                           "m"(DoubleFP[30]), "m"(DoubleFP[31]));
#endif
}

void JITResolver::CompilationCallback() {
  // Local space to save the registers
  uint64_t DoubleFP[32];
  uint64_t CC[3];
  uint64_t Globals[2];

  SaveRegisters(DoubleFP, CC, Globals);
  ++CallbackCalls;

  uint64_t CameFrom = (uint64_t)(intptr_t)__builtin_return_address(0);
  uint64_t CameFrom1 = (uint64_t)(intptr_t)__builtin_return_address(1);
  int64_t Target = (int64_t)TheJITResolver->resolveFunctionReference(CameFrom);
  DEBUG(std::cerr << "In callback! Addr=0x" << std::hex << CameFrom << "\n");
  register int64_t returnAddr = 0;
#if defined(sparc) || defined(__sparc__) || defined(__sparcv9)
  __asm__ __volatile__ ("add %%i7, %%g0, %0" : "=r" (returnAddr) : );
  DEBUG(std::cerr << "Read i7 (return addr) = "
                  << std::hex << returnAddr << ", value: "
                  << std::hex << *(unsigned*)returnAddr << "\n");
#endif

  // If we can rewrite the ORIGINAL caller, we eliminate the whole need for a
  // trampoline function stub!!
  unsigned OrigCallInst = *((unsigned*)(intptr_t)CameFrom1);
  int64_t OrigTarget = (Target-CameFrom1) >> 2;
  if ((OrigCallInst & (1 << 30)) && 
      (OrigTarget <= (1 << 30) && OrigTarget >= -(1 << 30)))
  {
    // The original call instruction was CALL <immed>, which means we can
    // overwrite it directly, since the offset will fit into 30 bits
    MachineInstr *C = BuildMI(V9::CALL, 1).addSImm(OrigTarget);
    *((unsigned*)(intptr_t)CameFrom1)=TheJITResolver->getBinaryCodeForInstr(*C);
    delete C;
    ++OverwrittenCalls;
  } else {
    ++UnmodifiedCalls;
  }

  // Rewrite the call target so that we don't fault every time we execute it.
  //

  static const unsigned o6 = SparcV9IntRegClass::o6;

  // Subtract enough to overwrite up to the 'save' instruction
  // This depends on whether we made a short call (1 instruction) or the
  // farCall (7 instructions)
  uint64_t Offset = (LazyCallFlavor[CameFrom] == ShortCall) ? 4 : 28;
  uint64_t CodeBegin = CameFrom - Offset;

  // FIXME FIXME FIXME FIXME: __builtin_frame_address doesn't work if frame
  // pointer elimination has been performed.  Having a variable sized alloca
  // disables frame pointer elimination currently, even if it's dead.  This is
  // a gross hack.
  alloca(42+Offset);
  // FIXME FIXME FIXME FIXME
  
  // Make sure that what we're about to overwrite is indeed "save"
  MachineInstr *SV =BuildMI(V9::SAVEi, 3).addReg(o6).addSImm(-192).addReg(o6);
  unsigned SaveInst = TheJITResolver->getBinaryCodeForInstr(*SV);
  delete SV;
  unsigned CodeInMem = *(unsigned*)(intptr_t)CodeBegin;
  if (CodeInMem != SaveInst) {
    std::cerr << "About to overwrite smthg not a save instr!";
    abort();
  }
  // Overwrite it
  TheJITResolver->insertJumpAtAddr(Target, CodeBegin);

  // Flush the I-Cache: FLUSH clears out a doubleword at a given address
  // Self-modifying code MUST clear out the I-Cache to be portable
#if defined(sparc) || defined(__sparc__) || defined(__sparcv9)
  for (int i = -Offset, e = 32-((int64_t)Offset); i < e; i += 8)
    __asm__ __volatile__ ("flush %%i7 + %0" : : "r" (i));
#endif

  // Change the return address to re-execute the restore, then the jump.
  DEBUG(std::cerr << "Callback returning to: 0x"
                  << std::hex << (CameFrom-Offset-12) << "\n");
#if defined(sparc) || defined(__sparc__) || defined(__sparcv9)
  __asm__ __volatile__ ("sub %%i7, %0, %%i7" : : "r" (Offset+12));
#endif

  RestoreRegisters(DoubleFP, CC, Globals);
}

/// emitStubForFunction - This method is used by the JIT when it needs to emit
/// the address of a function for a function whose code has not yet been
/// generated.  In order to do this, it generates a stub which jumps to the lazy
/// function compiler, which will eventually get fixed to call the function
/// directly.
///
uint64_t JITResolver::emitStubForFunction(Function *F) {
  MCE.startFunctionStub(*F, 44);

  DEBUG(std::cerr << "Emitting stub at addr: 0x" 
                  << std::hex << MCE.getCurrentPCValue() << "\n");

  unsigned o6 = SparcV9IntRegClass::o6, g0 = SparcV9IntRegClass::g0;

  // restore %g0, 0, %g0
  MachineInstr *R = BuildMI(V9::RESTOREi, 3).addMReg(g0).addSImm(0)
                                            .addMReg(g0, MachineOperand::Def);
  SparcV9.emitWord(SparcV9.getBinaryCodeForInstr(*R));
  delete R;

  // save %sp, -192, %sp
  MachineInstr *SV = BuildMI(V9::SAVEi, 3).addReg(o6).addSImm(-192).addReg(o6);
  SparcV9.emitWord(SparcV9.getBinaryCodeForInstr(*SV));
  delete SV;

  int64_t CurrPC = MCE.getCurrentPCValue();
  int64_t Addr = (int64_t)addFunctionReference(CurrPC, F);
  int64_t CallTarget = (Addr-CurrPC) >> 2;
  if (CallTarget >= (1 << 29) || CallTarget <= -(1 << 29)) {
    // Since this is a far call, the actual address of the call is shifted
    // by the number of instructions it takes to calculate the exact address
    deleteFunctionReference(CurrPC);
    SparcV9.emitFarCall(Addr, F);
  } else {
    // call CallTarget              ;; invoke the callback
    MachineInstr *Call = BuildMI(V9::CALL, 1).addSImm(CallTarget);
    SparcV9.emitWord(SparcV9.getBinaryCodeForInstr(*Call));
    delete Call;
  
    // nop                          ;; call delay slot
    MachineInstr *Nop = BuildMI(V9::NOP, 0);
    SparcV9.emitWord(SparcV9.getBinaryCodeForInstr(*Nop));
    delete Nop;

    addCallFlavor(CurrPC, ShortCall);
  }

  SparcV9.emitWord(0xDEADBEEF); // marker so that we know it's really a stub
  return (intptr_t)MCE.finishFunctionStub(*F)+4; /* 1 instr past the restore */
}

SparcV9CodeEmitter::SparcV9CodeEmitter(TargetMachine &tm,
                                       MachineCodeEmitter &M): TM(tm), MCE(M)
{
  TheJITResolver = new JITResolver(*this, M);
}

SparcV9CodeEmitter::~SparcV9CodeEmitter() {
  delete TheJITResolver;
}

void SparcV9CodeEmitter::emitWord(unsigned Val) {
  MCE.emitWord(Val);
}

unsigned 
SparcV9CodeEmitter::getRealRegNum(unsigned fakeReg,
                                  MachineInstr &MI) {
  const SparcV9RegInfo &RI = *TM.getRegInfo();
  unsigned regClass, regType = RI.getRegType(fakeReg);
  // At least map fakeReg into its class
  fakeReg = RI.getClassRegNum(fakeReg, regClass);

  switch (regClass) {
  case SparcV9RegInfo::IntRegClassID: {
    // SparcV9 manual, p31
    static const unsigned IntRegMap[] = {
      // "o0", "o1", "o2", "o3", "o4", "o5",       "o7",
      8, 9, 10, 11, 12, 13, 15,
      // "l0", "l1", "l2", "l3", "l4", "l5", "l6", "l7",
      16, 17, 18, 19, 20, 21, 22, 23,
      // "i0", "i1", "i2", "i3", "i4", "i5", "i6", "i7",
      24, 25, 26, 27, 28, 29, 30, 31,
      // "g0", "g1", "g2", "g3", "g4", "g5", "g6", "g7", 
      0, 1, 2, 3, 4, 5, 6, 7,
      // "o6"
      14
    }; 
 
    return IntRegMap[fakeReg];
    break;
  }
  case SparcV9RegInfo::FloatRegClassID: {
    DEBUG(std::cerr << "FP reg: " << fakeReg << "\n");
    if (regType == SparcV9RegInfo::FPSingleRegType) {
      // only numbered 0-31, hence can already fit into 5 bits (and 6)
      DEBUG(std::cerr << "FP single reg, returning: " << fakeReg << "\n");
    } else if (regType == SparcV9RegInfo::FPDoubleRegType) {
      // FIXME: This assumes that we only have 5-bit register fields!
      // From SparcV9 Manual, page 40.
      // The bit layout becomes: b[4], b[3], b[2], b[1], b[5]
      fakeReg |= (fakeReg >> 5) & 1;
      fakeReg &= 0x1f;
      DEBUG(std::cerr << "FP double reg, returning: " << fakeReg << "\n");      
    }
    return fakeReg;
  }
  case SparcV9RegInfo::IntCCRegClassID: {
    /*                                   xcc, icc, ccr */
    static const unsigned IntCCReg[] = {  6,   4,   2 };
    
    assert(fakeReg < sizeof(IntCCReg)/sizeof(IntCCReg[0])
             && "CC register out of bounds for IntCCReg map");      
    DEBUG(std::cerr << "IntCC reg: " << IntCCReg[fakeReg] << "\n");
    return IntCCReg[fakeReg];
  }
  case SparcV9RegInfo::FloatCCRegClassID: {
    /* These are laid out %fcc0 - %fcc3 => 0 - 3, so are correct */
    DEBUG(std::cerr << "FP CC reg: " << fakeReg << "\n");
    return fakeReg;
  }
  default:
    assert(0 && "Invalid unified register number in getRealRegNum");
    return fakeReg;
  }
}


// WARNING: if the call used the delay slot to do meaningful work, that's not
// being accounted for, and the behavior will be incorrect!!
inline void SparcV9CodeEmitter::emitFarCall(uint64_t Target, Function *F) {
  static const unsigned o6 = SparcV9IntRegClass::o6,
      o7 = SparcV9IntRegClass::o7, g0 = SparcV9IntRegClass::g0,
      g1 = SparcV9IntRegClass::g1, g5 = SparcV9IntRegClass::g5;

  MachineInstr* BinaryCode[] = {
    //
    // Get address to branch into %g1, using %g5 as a temporary
    //
    // sethi %uhi(Target), %g5   ;; get upper 22 bits of Target into %g5
    BuildMI(V9::SETHI, 2).addSImm(Target >> 42).addReg(g5),
    // or %g5, %ulo(Target), %g5 ;; get 10 lower bits of upper word into %1
    BuildMI(V9::ORi, 3).addReg(g5).addSImm((Target >> 32) & 0x03ff).addReg(g5),
    // sllx %g5, 32, %g5         ;; shift those 10 bits to the upper word
    BuildMI(V9::SLLXi6, 3).addReg(g5).addSImm(32).addReg(g5),
    // sethi %hi(Target), %g1    ;; extract bits 10-31 into the dest reg
    BuildMI(V9::SETHI, 2).addSImm((Target >> 10) & 0x03fffff).addReg(g1),
    // or %g5, %g1, %g1          ;; get upper word (in %g5) into %g1
    BuildMI(V9::ORr, 3).addReg(g5).addReg(g1).addReg(g1),
    // or %g1, %lo(Target), %g1  ;; get lowest 10 bits of Target into %g1
    BuildMI(V9::ORi, 3).addReg(g1).addSImm(Target & 0x03ff).addReg(g1),
    // jmpl %g1, %g0, %o7        ;; indirect call on %g1
    BuildMI(V9::JMPLRETr, 3).addReg(g1).addReg(g0).addReg(o7),
    // nop                       ;; delay slot
    BuildMI(V9::NOP, 0)
  };

  for (unsigned i=0, e=sizeof(BinaryCode)/sizeof(BinaryCode[0]); i!=e; ++i) {
    // This is where we save the return address in the LazyResolverMap!!
    if (i == 6 && F != 0) { // Do this right before the JMPL
      uint64_t CurrPC = MCE.getCurrentPCValue();
      TheJITResolver->addFunctionReference(CurrPC, F);
      // Remember that this is a far call, to subtract appropriate offset later
      TheJITResolver->addCallFlavor(CurrPC, JITResolver::FarCall);
    }

    emitWord(getBinaryCodeForInstr(*BinaryCode[i]));
    delete BinaryCode[i];
  }
}

void SparcV9JITInfo::replaceMachineCodeForFunction (void *Old, void *New) {
  assert (TheJITResolver &&
	"Can only call replaceMachineCodeForFunction from within JIT");
  uint64_t Target = (uint64_t)(intptr_t)New;
  uint64_t CodeBegin = (uint64_t)(intptr_t)Old;
  TheJITResolver->insertJumpAtAddr(Target, CodeBegin);
}

int64_t SparcV9CodeEmitter::getMachineOpValue(MachineInstr &MI,
                                              MachineOperand &MO) {
  int64_t rv = 0; // Return value; defaults to 0 for unhandled cases
                  // or things that get fixed up later by the JIT.
  if (MO.isPCRelativeDisp()) {
    DEBUG(std::cerr << "PCRelativeDisp: ");
    Value *V = MO.getVRegValue();
    if (BasicBlock *BB = dyn_cast<BasicBlock>(V)) {
      DEBUG(std::cerr << "Saving reference to BB (VReg)\n");
      unsigned* CurrPC = (unsigned*)(intptr_t)MCE.getCurrentPCValue();
      BBRefs.push_back(std::make_pair(BB, std::make_pair(CurrPC, &MI)));
    } else if (const ConstantInt *CI = dyn_cast<ConstantInt>(V)) {
      // The real target of the branch is CI = PC + (rv * 4)
      // So undo that: give the instruction (CI - PC) / 4
      rv = (CI->getRawValue() - MCE.getCurrentPCValue()) / 4;
    } else if (GlobalValue *GV = dyn_cast<GlobalValue>(V)) {
      // same as MO.isGlobalAddress()
      DEBUG(std::cerr << "GlobalValue: ");
      // external function calls, etc.?
      if (Function *F = dyn_cast<Function>(GV)) {
        DEBUG(std::cerr << "Function: ");
        // NOTE: This results in stubs being generated even for
        // external, native functions, which is not optimal. See PR103.
        rv = (int64_t)MCE.getGlobalValueAddress(F);
        if (rv == 0) {
          DEBUG(std::cerr << "not yet generated\n");
          // Function has not yet been code generated!
          TheJITResolver->addFunctionReference(MCE.getCurrentPCValue(), F);
          // Delayed resolution...
          rv = TheJITResolver->getLazyResolver(F);
        } else {
          DEBUG(std::cerr << "already generated: 0x" << std::hex << rv << "\n");
        }
      } else {
        rv = (int64_t)MCE.getGlobalValueAddress(GV);
        DEBUG(std::cerr << "Global addr: 0x" << std::hex << rv << "\n");
      }
      // The real target of the call is Addr = PC + (rv * 4)
      // So undo that: give the instruction (Addr - PC) / 4
      if (MI.getOpcode() == V9::CALL) {
        int64_t CurrPC = MCE.getCurrentPCValue();
        DEBUG(std::cerr << "rv addr: 0x" << std::hex << rv << "\n"
                        << "curr PC: 0x" << std::hex << CurrPC << "\n");
        int64_t CallInstTarget = (rv - CurrPC) >> 2;
        if (CallInstTarget >= (1<<29) || CallInstTarget <= -(1<<29)) {
          DEBUG(std::cerr << "Making far call!\n");
          // address is out of bounds for the 30-bit call,
          // make an indirect jump-and-link
          emitFarCall(rv);
          // this invalidates the instruction so that the call with an incorrect
          // address will not be emitted
          rv = 0; 
        } else {
          // The call fits into 30 bits, so just return the corrected address
          rv = CallInstTarget;
        }
        DEBUG(std::cerr << "returning addr: 0x" << rv << "\n");
      }
    } else {
      std::cerr << "ERROR: PC relative disp unhandled:" << MO << "\n";
      abort();
    }
  } else if (MO.isRegister() || MO.getType() == MachineOperand::MO_CCRegister)
  {
    // This is necessary because the SparcV9 backend doesn't actually lay out
    // registers in the real fashion -- it skips those that it chooses not to
    // allocate, i.e. those that are the FP, SP, etc.
    unsigned fakeReg = MO.getReg();
    unsigned realRegByClass = getRealRegNum(fakeReg, MI);
    DEBUG(std::cerr << MO << ": Reg[" << std::dec << fakeReg << "] => "
                    << realRegByClass << " (LLC: " 
                    << TM.getRegInfo()->getUnifiedRegName(fakeReg) << ")\n");
    rv = realRegByClass;
  } else if (MO.isImmediate()) {
    rv = MO.getImmedValue();
    DEBUG(std::cerr << "immed: " << rv << "\n");
  } else if (MO.isGlobalAddress()) {
    DEBUG(std::cerr << "GlobalAddress: not PC-relative\n");
    rv = (int64_t)
      (intptr_t)getGlobalAddress(cast<GlobalValue>(MO.getVRegValue()),
                                 MI, MO.isPCRelative());
  } else if (MO.isMachineBasicBlock()) {
    // Duplicate code of the above case for VirtualRegister, BasicBlock... 
    // It should really hit this case, but SparcV9 backend uses VRegs instead
    DEBUG(std::cerr << "Saving reference to MBB\n");
    const BasicBlock *BB = MO.getMachineBasicBlock()->getBasicBlock();
    unsigned* CurrPC = (unsigned*)(intptr_t)MCE.getCurrentPCValue();
    BBRefs.push_back(std::make_pair(BB, std::make_pair(CurrPC, &MI)));
  } else if (MO.isExternalSymbol()) {
    // SparcV9 backend doesn't generate this (yet...)
    std::cerr << "ERROR: External symbol unhandled: " << MO << "\n";
    abort();
  } else if (MO.isFrameIndex()) {
    // SparcV9 backend doesn't generate this (yet...)
    int FrameIndex = MO.getFrameIndex();
    std::cerr << "ERROR: Frame index unhandled.\n";
    abort();
  } else if (MO.isConstantPoolIndex()) {
    unsigned Index = MO.getConstantPoolIndex();
    rv = MCE.getConstantPoolEntryAddress(Index);
  } else {
    std::cerr << "ERROR: Unknown type of MachineOperand: " << MO << "\n";
    abort();
  }

  // Finally, deal with the various bitfield-extracting functions that
  // are used in SPARC assembly. (Some of these make no sense in combination
  // with some of the above; we'll trust that the instruction selector
  // will not produce nonsense, and not check for valid combinations here.)
  if (MO.isLoBits32()) {          // %lo(val) == %lo() in SparcV9 ABI doc
    return rv & 0x03ff;
  } else if (MO.isHiBits32()) {   // %lm(val) == %hi() in SparcV9 ABI doc
    return (rv >> 10) & 0x03fffff;
  } else if (MO.isLoBits64()) {   // %hm(val) == %ulo() in SparcV9 ABI doc
    return (rv >> 32) & 0x03ff;
  } else if (MO.isHiBits64()) {   // %hh(val) == %uhi() in SparcV9 ABI doc
    return rv >> 42;
  } else {                        // (unadorned) val
    return rv;
  }
}

unsigned SparcV9CodeEmitter::getValueBit(int64_t Val, unsigned bit) {
  Val >>= bit;
  return (Val & 1);
}

bool SparcV9CodeEmitter::runOnMachineFunction(MachineFunction &MF) {
  MCE.startFunction(MF);
  DEBUG(std::cerr << "Starting function " << MF.getFunction()->getName()
            << ", address: " << "0x" << std::hex 
            << (long)MCE.getCurrentPCValue() << "\n");

  MCE.emitConstantPool(MF.getConstantPool());
  for (MachineFunction::iterator I = MF.begin(), E = MF.end(); I != E; ++I)
    emitBasicBlock(*I);
  MCE.finishFunction(MF);

  DEBUG(std::cerr << "Finishing fn " << MF.getFunction()->getName() << "\n");

  // Resolve branches to BasicBlocks for the entire function
  for (unsigned i = 0, e = BBRefs.size(); i != e; ++i) {
    long Location = BBLocations[BBRefs[i].first];
    unsigned *Ref = BBRefs[i].second.first;
    MachineInstr *MI = BBRefs[i].second.second;
    DEBUG(std::cerr << "Fixup @ " << std::hex << Ref << " to 0x" << Location
                    << " in instr: " << std::dec << *MI);
    for (unsigned ii = 0, ee = MI->getNumOperands(); ii != ee; ++ii) {
      MachineOperand &op = MI->getOperand(ii);
      if (op.isPCRelativeDisp()) {
        // the instruction's branch target is made such that it branches to
        // PC + (branchTarget * 4), so undo that arithmetic here:
        // Location is the target of the branch
        // Ref is the location of the instruction, and hence the PC
        int64_t branchTarget = (Location - (long)Ref) >> 2;
        // Save the flags.
        bool loBits32=false, hiBits32=false, loBits64=false, hiBits64=false;   
        if (op.isLoBits32()) { loBits32=true; }
        if (op.isHiBits32()) { hiBits32=true; }
        if (op.isLoBits64()) { loBits64=true; }
        if (op.isHiBits64()) { hiBits64=true; }
        MI->SetMachineOperandConst(ii, MachineOperand::MO_SignExtendedImmed,
                                   branchTarget);
        if (loBits32) { MI->setOperandLo32(ii); }
        else if (hiBits32) { MI->setOperandHi32(ii); }
        else if (loBits64) { MI->setOperandLo64(ii); }
        else if (hiBits64) { MI->setOperandHi64(ii); }
        DEBUG(std::cerr << "Rewrote BB ref: ");
        unsigned fixedInstr = SparcV9CodeEmitter::getBinaryCodeForInstr(*MI);
        MCE.emitWordAt (fixedInstr, Ref);
        break;
      }
    }
  }
  BBRefs.clear();
  BBLocations.clear();

  return false;
}

void SparcV9CodeEmitter::emitBasicBlock(MachineBasicBlock &MBB) {
  currBB = MBB.getBasicBlock();
  BBLocations[currBB] = MCE.getCurrentPCValue();
  for (MachineBasicBlock::iterator I = MBB.begin(), E = MBB.end(); I != E; ++I){
    unsigned binCode = getBinaryCodeForInstr(*I);
    if (binCode == (1 << 30)) {
      // this is an invalid call: the addr is out of bounds. that means a code
      // sequence has already been emitted, and this is a no-op
      DEBUG(std::cerr << "Call suppressed: already emitted far call.\n");
    } else {
      emitWord(binCode);
    }
  }
}

void* SparcV9CodeEmitter::getGlobalAddress(GlobalValue *V, MachineInstr &MI,
                                           bool isPCRelative)
{
  if (isPCRelative) { // must be a call, this is a major hack!
    // Try looking up the function to see if it is already compiled!
    if (void *Addr = (void*)(intptr_t)MCE.getGlobalValueAddress(V)) {
      intptr_t CurByte = MCE.getCurrentPCValue();
      // The real target of the call is Addr = PC + (target * 4)
      // CurByte is the PC, Addr we just received
      return (void*) (((long)Addr - (long)CurByte) >> 2);
    } else {
      if (Function *F = dyn_cast<Function>(V)) {
        // Function has not yet been code generated!
        TheJITResolver->addFunctionReference(MCE.getCurrentPCValue(),
                                             cast<Function>(V));
        // Delayed resolution...
        return 
          (void*)(intptr_t)TheJITResolver->getLazyResolver(cast<Function>(V));
      } else {
        std::cerr << "Unhandled global: " << *V << "\n";
        abort();
      }
    }
  } else {
    return (void*)(intptr_t)MCE.getGlobalValueAddress(V);
  }
}

#include "SparcV9CodeEmitter.inc"

} // End llvm namespace

