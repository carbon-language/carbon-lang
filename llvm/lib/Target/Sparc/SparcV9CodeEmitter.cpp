//===-- SparcV9CodeEmitter.cpp -  --------===//
//
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
#include "Support/Statistic.h"
#include "Support/hash_set"
#include "SparcInternals.h"
#include "SparcV9CodeEmitter.h"

bool UltraSparc::addPassesToEmitMachineCode(PassManager &PM,
                                            MachineCodeEmitter &MCE) {
  MachineCodeEmitter *M = &MCE;
  DEBUG(M = MachineCodeEmitter::createFilePrinterEmitter(MCE));
  PM.add(new SparcV9CodeEmitter(*this, *M));
  PM.add(createMachineCodeDestructionPass()); // Free stuff no longer needed
  return false;
}

namespace {
  class JITResolver {
    SparcV9CodeEmitter &SparcV9;
    MachineCodeEmitter &MCE;

    // LazyCodeGenMap - Keep track of call sites for functions that are to be
    // lazily resolved.
    std::map<uint64_t, Function*> LazyCodeGenMap;

    // LazyResolverMap - Keep track of the lazy resolver created for a
    // particular function so that we can reuse them if necessary.
    std::map<Function*, uint64_t> LazyResolverMap;
  public:
    JITResolver(SparcV9CodeEmitter &V9,
                MachineCodeEmitter &mce) : SparcV9(V9), MCE(mce) {}
    uint64_t getLazyResolver(Function *F);
    uint64_t addFunctionReference(uint64_t Address, Function *F);

    // Utility functions for accessing data from static callback
    uint64_t getCurrentPCValue() {
      return MCE.getCurrentPCValue();
    }
    unsigned getBinaryCodeForInstr(MachineInstr &MI) {
      return SparcV9.getBinaryCodeForInstr(MI);
    }

    inline uint64_t insertFarJumpAtAddr(int64_t Value, uint64_t Addr);

  private:
    uint64_t emitStubForFunction(Function *F);
    static void CompilationCallback();
    uint64_t resolveFunctionReference(uint64_t RetAddr);

  };

  JITResolver *TheJITResolver;
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
  
//std::cerr << "Getting lazy resolver for : " << ((Value*)F)->getName() << "\n";

  uint64_t Stub = emitStubForFunction(F);
  LazyResolverMap.insert(I, std::make_pair(F, Stub));
  return Stub;
}

uint64_t JITResolver::insertFarJumpAtAddr(int64_t Target, uint64_t Addr) {

  static const unsigned i1 = SparcIntRegClass::i1, i2 = SparcIntRegClass::i2,
    i7 = SparcIntRegClass::i7,
    o6 = SparcIntRegClass::o6, g0 = SparcIntRegClass::g0;

  // 
  // Save %i1, %i2 to the stack so we can form a 64-bit constant in %i2
  // 

  // stx %i1, [%sp + 2119]       ;; save %i1 to the stack, used as temp
  MachineInstr *STX = BuildMI(V9::STXi, 3).addReg(i1).addReg(o6).addSImm(2119);
  *((unsigned*)(intptr_t)Addr) = getBinaryCodeForInstr(*STX);
  delete STX;
  Addr += 4;

  // stx %i2, [%sp + 2127]       ;; save %i2 to the stack
  STX = BuildMI(V9::STXi, 3).addReg(i2).addReg(o6).addSImm(2127);
  *((unsigned*)(intptr_t)Addr) = getBinaryCodeForInstr(*STX);
  delete STX;
  Addr += 4;

  //
  // Get address to branch into %i2, using %i1 as a temporary
  //

  // sethi %uhi(Target), %i1   ;; get upper 22 bits of Target into %i1
  MachineInstr *SH = BuildMI(V9::SETHI, 2).addSImm(Target >> 42).addReg(i1);
  *((unsigned*)(intptr_t)Addr) = getBinaryCodeForInstr(*SH);
  delete SH;
  Addr += 4;

  // or %i1, %ulo(Target), %i1 ;; get 10 lower bits of upper word into %1
  MachineInstr *OR = BuildMI(V9::ORi, 3)
    .addReg(i1).addSImm((Target >> 32) & 0x03ff).addReg(i1);
  *((unsigned*)(intptr_t)Addr) = getBinaryCodeForInstr(*OR);
  delete OR;
  Addr += 4;

  // sllx %i1, 32, %i1            ;; shift those 10 bits to the upper word
  MachineInstr *SL = BuildMI(V9::SLLXi6, 3).addReg(i1).addSImm(32).addReg(i1);
  *((unsigned*)(intptr_t)Addr) = getBinaryCodeForInstr(*SL);
  delete SL;
  Addr += 4;

  // sethi %hi(Target), %i2    ;; extract bits 10-31 into the dest reg
  SH = BuildMI(V9::SETHI, 2).addSImm((Target >> 10) & 0x03fffff).addReg(i2);
  *((unsigned*)(intptr_t)Addr) = getBinaryCodeForInstr(*SH);
  delete SH;
  Addr += 4;

  // or %i1, %i2, %i2             ;; get upper word (in %i1) into %i2
  OR = BuildMI(V9::ORr, 3).addReg(i1).addReg(i2).addReg(i2);
  *((unsigned*)(intptr_t)Addr) = getBinaryCodeForInstr(*OR);
  delete OR;
  Addr += 4;

  // or %i2, %lo(Target), %i2  ;; get lowest 10 bits of Target into %i2
  OR = BuildMI(V9::ORi, 3).addReg(i2).addSImm(Target & 0x03ff).addReg(i2);
  *((unsigned*)(intptr_t)Addr) = getBinaryCodeForInstr(*OR);
  delete OR;
  Addr += 4;

  // ldx [%sp + 2119], %i1       ;; restore %i1 -> 2119 = BIAS(2047) + 72
  MachineInstr *LDX = BuildMI(V9::LDXi, 3).addReg(o6).addSImm(2119).addReg(i1);
  *((unsigned*)(intptr_t)Addr) = getBinaryCodeForInstr(*LDX);
  delete LDX;
  Addr += 4;

  // jmpl %i2, %g0, %g0          ;; indirect branch on %i2
  MachineInstr *J = BuildMI(V9::JMPLRETr, 3).addReg(i2).addReg(g0).addReg(g0);
  *((unsigned*)(intptr_t)Addr) = getBinaryCodeForInstr(*J);
  delete J;
  Addr += 4;

  // ldx [%sp + 2127], %i2       ;; restore %i2 -> 2127 = BIAS(2047) + 80
  LDX = BuildMI(V9::LDXi, 3).addReg(o6).addSImm(2127).addReg(i2);
  *((unsigned*)(intptr_t)Addr) = getBinaryCodeForInstr(*LDX);
  delete LDX;
  Addr += 4;

  return Addr;
}

void JITResolver::CompilationCallback() {
  uint64_t CameFrom = (uint64_t)(intptr_t)__builtin_return_address(0);
  int64_t Target = (int64_t)TheJITResolver->resolveFunctionReference(CameFrom);
  DEBUG(std::cerr << "In callback! Addr=0x" << std::hex << CameFrom << "\n");

  // Rewrite the call target... so that we don't fault every time we execute
  // the call.
#if 0
  int64_t RealCallTarget = (int64_t)
    ((NewVal - TheJITResolver->getCurrentPCValue()) >> 4);
  if (RealCallTarget >= (1<<22) || RealCallTarget <= -(1<<22)) {
    std::cerr << "Address out of bounds for 22bit BA: " << RealCallTarget<<"\n";
    abort();
  }
#endif

  //uint64_t CurrPC    = TheJITResolver->getCurrentPCValue();
  // we will insert 9 instructions before we do the actual jump
  //int64_t NewTarget  = (NewVal - 9*4 - InstAddr) >> 2;

  static const unsigned i1 = SparcIntRegClass::i1, i2 = SparcIntRegClass::i2,
    i7 = SparcIntRegClass::i7, o6 = SparcIntRegClass::o6,
    o7 = SparcIntRegClass::o7, g0 = SparcIntRegClass::g0;

  // Subtract 4 to overwrite the 'save' that's there now
  uint64_t InstAddr = CameFrom-4;

  InstAddr = TheJITResolver->insertFarJumpAtAddr(Target, InstAddr);

  // CODE SHOULD NEVER GO PAST THIS LOAD!! The real function should return to
  // the original caller, not here!!

  // FIXME: add call 0 to make sure?!?

  // =============== THE REAL STUB ENDS HERE =========================

  // What follows below is one-time restore code, because this callback may be
  // changing registers in unpredictible ways. However, since it is executed
  // only once per function (after the function is resolved, the callback is no
  // longer in the path), this has to be done only once.
  //
  // Thus, it is after the regular stub code. The call back returns to THIS
  // point, but every other call to the target function will execute the code
  // above. Hence, this code is one-time use.

  uint64_t OneTimeRestore = InstAddr;

  // restore %g0, 0, %g0
  //MachineInstr *R = BuildMI(V9::RESTOREi, 3).addMReg(g0).addSImm(0)
  //                                          .addMReg(g0, MOTy::Def);
  //*((unsigned*)(intptr_t)InstAddr)=TheJITResolver->getBinaryCodeForInstr(*R);
  //delete R;

  // FIXME: BuildMI() above crashes. Encode the instruction directly.
  // restore %g0, 0, %g0
  *((unsigned*)(intptr_t)InstAddr) = 0x81e82000U;
  InstAddr += 4;  

  InstAddr = TheJITResolver->insertFarJumpAtAddr(Target, InstAddr);

  // FIXME: if the target function is close enough to fit into the 19bit disp of
  // BA, we should use this version, as its much cheaper to generate.
  /*
  MachineInstr *MI = BuildMI(V9::BA, 1).addSImm(RealCallTarget);
  *((unsigned*)(intptr_t)InstAddr) = TheJITResolver->getBinaryCodeForInstr(*MI);
  delete MI;
  InstAddr += 4;

  // Add another NOP
  MachineInstr *Nop = BuildMI(V9::NOP, 0);
  *((unsigned*)(intptr_t)InstAddr)=TheJITResolver->getBinaryCodeForInstr(*Nop);
  delete Nop;
  InstAddr += 4;

  MachineInstr *BA = BuildMI(V9::BA, 1).addSImm(RealCallTarget-2);
  *((unsigned*)(intptr_t)InstAddr) = TheJITResolver->getBinaryCodeForInstr(*BA);
  delete BA;
  */

  // Change the return address to reexecute the call instruction...
  // The return address is really %o7, but will disappear after this function
  // returns, and the register windows are rotated away.
#if defined(sparc) || defined(__sparc__) || defined(__sparcv9)
  __asm__ __volatile__ ("or %%g0, %0, %%i7" : : "r" (OneTimeRestore-8));
#endif
}

/// emitStubForFunction - This method is used by the JIT when it needs to emit
/// the address of a function for a function whose code has not yet been
/// generated.  In order to do this, it generates a stub which jumps to the lazy
/// function compiler, which will eventually get fixed to call the function
/// directly.
///
uint64_t JITResolver::emitStubForFunction(Function *F) {
  MCE.startFunctionStub(*F, 6);

  DEBUG(std::cerr << "Emitting stub at addr: 0x" 
                  << std::hex << MCE.getCurrentPCValue() << "\n");

  unsigned o6 = SparcIntRegClass::o6;
  // save %sp, -192, %sp
  MachineInstr *SV = BuildMI(V9::SAVEi, 3).addReg(o6).addSImm(-192).addReg(o6);
  SparcV9.emitWord(SparcV9.getBinaryCodeForInstr(*SV));
  delete SV;

  int64_t CurrPC = MCE.getCurrentPCValue();
  int64_t Addr = (int64_t)addFunctionReference(CurrPC, F);

  int64_t CallTarget = (Addr-CurrPC) >> 2;
  if (CallTarget >= (1 << 30) || CallTarget <= -(1 << 30)) {
    std::cerr << "Call target beyond 30 bit limit of CALL: " 
              << CallTarget << "\n";
    abort();
  }
  // call CallTarget              ;; invoke the callback
  MachineInstr *Call = BuildMI(V9::CALL, 1).addSImm(CallTarget);
  SparcV9.emitWord(SparcV9.getBinaryCodeForInstr(*Call));
  delete Call;
  
  // nop                          ;; call delay slot
  MachineInstr *Nop = BuildMI(V9::NOP, 0);
  SparcV9.emitWord(SparcV9.getBinaryCodeForInstr(*Nop));
  delete Nop;

  SparcV9.emitWord(0xDEADBEEF); // marker so that we know it's really a stub
  return (intptr_t)MCE.finishFunctionStub(*F);
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
  // Output the constant in big endian byte order...
  unsigned byteVal;
  for (int i = 3; i >= 0; --i) {
    byteVal = Val >> 8*i;
    MCE.emitByte(byteVal & 255);
  }
}

unsigned 
SparcV9CodeEmitter::getRealRegNum(unsigned fakeReg,
                                         MachineInstr &MI) {
  const TargetRegInfo &RI = TM.getRegInfo();
  unsigned regClass, regType = RI.getRegType(fakeReg);
  // At least map fakeReg into its class
  fakeReg = RI.getClassRegNum(fakeReg, regClass);

  switch (regClass) {
  case UltraSparcRegInfo::IntRegClassID: {
    // Sparc manual, p31
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
  case UltraSparcRegInfo::FloatRegClassID: {
    DEBUG(std::cerr << "FP reg: " << fakeReg << "\n");
    if (regType == UltraSparcRegInfo::FPSingleRegType) {
      // only numbered 0-31, hence can already fit into 5 bits (and 6)
      DEBUG(std::cerr << "FP single reg, returning: " << fakeReg << "\n");
    } else if (regType == UltraSparcRegInfo::FPDoubleRegType) {
      // FIXME: This assumes that we only have 5-bit register fiels!
      // From Sparc Manual, page 40.
      // The bit layout becomes: b[4], b[3], b[2], b[1], b[5]
      fakeReg |= (fakeReg >> 5) & 1;
      fakeReg &= 0x1f;
      DEBUG(std::cerr << "FP double reg, returning: " << fakeReg << "\n");      
    }
    return fakeReg;
  }
  case UltraSparcRegInfo::IntCCRegClassID: {
    /*                                   xcc, icc, ccr */
    static const unsigned IntCCReg[] = {  6,   4,   2 };
    
    assert(fakeReg < sizeof(IntCCReg)/sizeof(IntCCReg[0])
             && "CC register out of bounds for IntCCReg map");      
    DEBUG(std::cerr << "IntCC reg: " << IntCCReg[fakeReg] << "\n");
    return IntCCReg[fakeReg];
  }
  case UltraSparcRegInfo::FloatCCRegClassID: {
    /* These are laid out %fcc0 - %fcc3 => 0 - 3, so are correct */
    DEBUG(std::cerr << "FP CC reg: " << fakeReg << "\n");
    return fakeReg;
  }
  default:
    assert(0 && "Invalid unified register number in getRegType");
    return fakeReg;
  }
}


// WARNING: if the call used the delay slot to do meaningful work, that's not
// being accounted for, and the behavior will be incorrect!!
inline void SparcV9CodeEmitter::emitFarCall(uint64_t Target) {
  static const unsigned i1 = SparcIntRegClass::i1, i2 = SparcIntRegClass::i2,
    i7 = SparcIntRegClass::i7,
    o6 = SparcIntRegClass::o6, g0 = SparcIntRegClass::g0;

  // 
  // Save %i1, %i2 to the stack so we can form a 64-bit constant in %i2
  // 

  // stx %i1, [%sp + 2119]       ;; save %i1 to the stack, used as temp
  MachineInstr *STX = BuildMI(V9::STXi, 3).addReg(i1).addReg(o6).addSImm(2119);
  emitWord(getBinaryCodeForInstr(*STX));
  delete STX;

  // stx %i2, [%sp + 2127]       ;; save %i2 to the stack
  STX = BuildMI(V9::STXi, 3).addReg(i2).addReg(o6).addSImm(2127);
  emitWord(getBinaryCodeForInstr(*STX));
  delete STX;

  //
  // Get address to branch into %i2, using %i1 as a temporary
  //

  // sethi %uhi(Target), %i1   ;; get upper 22 bits of Target into %i1
  MachineInstr *SH = BuildMI(V9::SETHI, 2).addSImm(Target >> 42).addReg(i1);
  emitWord(getBinaryCodeForInstr(*SH));
  delete SH;

  // or %i1, %ulo(Target), %i1 ;; get 10 lower bits of upper word into %1
  MachineInstr *OR = BuildMI(V9::ORi, 3)
    .addReg(i1).addSImm((Target >> 32) & 0x03ff).addReg(i1);
  emitWord(getBinaryCodeForInstr(*OR));
  delete OR;

  // sllx %i1, 32, %i1            ;; shift those 10 bits to the upper word
  MachineInstr *SL = BuildMI(V9::SLLXi6, 3).addReg(i1).addSImm(32).addReg(i1);
  emitWord(getBinaryCodeForInstr(*SL));
  delete SL;

  // sethi %hi(Target), %i2    ;; extract bits 10-31 into the dest reg
  SH = BuildMI(V9::SETHI, 2).addSImm((Target >> 10) & 0x03fffff).addReg(i2);
  emitWord(getBinaryCodeForInstr(*SH));
  delete SH;

  // or %i1, %i2, %i2             ;; get upper word (in %i1) into %i2
  OR = BuildMI(V9::ORr, 3).addReg(i1).addReg(i2).addReg(i2);
  emitWord(getBinaryCodeForInstr(*OR));
  delete OR;

  // or %i2, %lo(Target), %i2  ;; get lowest 10 bits of Target into %i2
  OR = BuildMI(V9::ORi, 3).addReg(i2).addSImm(Target & 0x03ff).addReg(i2);
  emitWord(getBinaryCodeForInstr(*OR));
  delete OR;

  // ldx [%sp + 2119], %i1       ;; restore %i1 -> 2119 = BIAS(2047) + 72
  MachineInstr *LDX = BuildMI(V9::LDXi, 3).addReg(o6).addSImm(2119).addReg(i1);
  emitWord(getBinaryCodeForInstr(*LDX));
  delete LDX;

  // jmpl %i2, %g0, %07          ;; indirect call on %i2
  MachineInstr *J = BuildMI(V9::JMPLRETr, 3).addReg(i2).addReg(g0).addReg(07);
  emitWord(getBinaryCodeForInstr(*J));
  delete J;

  // ldx [%sp + 2127], %i2       ;; restore %i2 -> 2127 = BIAS(2047) + 80
  LDX = BuildMI(V9::LDXi, 3).addReg(o6).addSImm(2127).addReg(i2);
  emitWord(getBinaryCodeForInstr(*LDX));
  delete LDX;
}


int64_t SparcV9CodeEmitter::getMachineOpValue(MachineInstr &MI,
                                              MachineOperand &MO) {
  int64_t rv = 0; // Return value; defaults to 0 for unhandled cases
                  // or things that get fixed up later by the JIT.

  if (MO.isVirtualRegister()) {
    std::cerr << "ERROR: virtual register found in machine code.\n";
    abort();
  } else if (MO.isPCRelativeDisp()) {
    DEBUG(std::cerr << "PCRelativeDisp: ");
    Value *V = MO.getVRegValue();
    if (BasicBlock *BB = dyn_cast<BasicBlock>(V)) {
      DEBUG(std::cerr << "Saving reference to BB (VReg)\n");
      unsigned* CurrPC = (unsigned*)(intptr_t)MCE.getCurrentPCValue();
      BBRefs.push_back(std::make_pair(BB, std::make_pair(CurrPC, &MI)));
    } else if (const Constant *C = dyn_cast<Constant>(V)) {
      if (ConstantMap.find(C) != ConstantMap.end()) {
        rv = (int64_t)MCE.getConstantPoolEntryAddress(ConstantMap[C]);
        DEBUG(std::cerr << "const: 0x" << std::hex << rv << "\n");
      } else {
        std::cerr << "ERROR: constant not in map:" << MO << "\n";
        abort();
      }
    } else if (GlobalValue *GV = dyn_cast<GlobalValue>(V)) {
      // same as MO.isGlobalAddress()
      DEBUG(std::cerr << "GlobalValue: ");
      // external function calls, etc.?
      if (Function *F = dyn_cast<Function>(GV)) {
        DEBUG(std::cerr << "Function: ");
        if (F->isExternal()) {
          // Sparc backend broken: this MO should be `ExternalSymbol'
          rv = (int64_t)MCE.getGlobalValueAddress(F->getName());
        } else {
          rv = (int64_t)MCE.getGlobalValueAddress(F);
        }
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
        if (rv == 0) {
          if (Constant *C = ConstantPointerRef::get(GV)) {
            if (ConstantMap.find(C) != ConstantMap.end()) {
              rv = MCE.getConstantPoolEntryAddress(ConstantMap[C]);
            } else {
              std::cerr << "Constant: 0x" << std::hex << (intptr_t)C
                        << ", " << *V << " not found in ConstantMap!\n";
              abort();
            }
          }
        }
        DEBUG(std::cerr << "Global addr: " << rv << "\n");
      }
      // The real target of the call is Addr = PC + (rv * 4)
      // So undo that: give the instruction (Addr - PC) / 4
      if (MI.getOpcode() == V9::CALL) {
        int64_t CurrPC = MCE.getCurrentPCValue();
        DEBUG(std::cerr << "rv addr: 0x" << std::hex << rv << "\n"
                        << "curr PC: 0x" << CurrPC << "\n");
        int64_t CallInstTarget = (rv - CurrPC) >> 2;
        if (CallInstTarget >= (1<<29) || CallInstTarget <= -(1<<29)) {
          DEBUG(std::cerr << "Making far call!\n");
          // addresss is out of bounds for the 30-bit call,
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
  } else if (MO.isPhysicalRegister() ||
             MO.getType() == MachineOperand::MO_CCRegister)
  {
    // This is necessary because the Sparc backend doesn't actually lay out
    // registers in the real fashion -- it skips those that it chooses not to
    // allocate, i.e. those that are the FP, SP, etc.
    unsigned fakeReg = MO.getAllocatedRegNum();
    unsigned realRegByClass = getRealRegNum(fakeReg, MI);
    DEBUG(std::cerr << MO << ": Reg[" << std::dec << fakeReg << "] => "
                    << realRegByClass << " (LLC: " 
                    << TM.getRegInfo().getUnifiedRegName(fakeReg) << ")\n");
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
    // It should really hit this case, but Sparc backend uses VRegs instead
    DEBUG(std::cerr << "Saving reference to MBB\n");
    BasicBlock *BB = MO.getMachineBasicBlock()->getBasicBlock();
    unsigned* CurrPC = (unsigned*)(intptr_t)MCE.getCurrentPCValue();
    BBRefs.push_back(std::make_pair(BB, std::make_pair(CurrPC, &MI)));
  } else if (MO.isExternalSymbol()) {
    // Sparc backend doesn't generate this (yet...)
    std::cerr << "ERROR: External symbol unhandled: " << MO << "\n";
    abort();
  } else if (MO.isFrameIndex()) {
    // Sparc backend doesn't generate this (yet...)
    int FrameIndex = MO.getFrameIndex();
    std::cerr << "ERROR: Frame index unhandled.\n";
    abort();
  } else if (MO.isConstantPoolIndex()) {
    // Sparc backend doesn't generate this (yet...)
    std::cerr << "ERROR: Constant Pool index unhandled.\n";
    abort();
  } else {
    std::cerr << "ERROR: Unknown type of MachineOperand: " << MO << "\n";
    abort();
  }

  // Finally, deal with the various bitfield-extracting functions that
  // are used in SPARC assembly. (Some of these make no sense in combination
  // with some of the above; we'll trust that the instruction selector
  // will not produce nonsense, and not check for valid combinations here.)
  if (MO.opLoBits32()) {          // %lo(val) == %lo() in Sparc ABI doc
    return rv & 0x03ff;
  } else if (MO.opHiBits32()) {   // %lm(val) == %hi() in Sparc ABI doc
    return (rv >> 10) & 0x03fffff;
  } else if (MO.opLoBits64()) {   // %hm(val) == %ulo() in Sparc ABI doc
    return (rv >> 32) & 0x03ff;
  } else if (MO.opHiBits64()) {   // %hh(val) == %uhi() in Sparc ABI doc
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

  // The Sparc backend does not use MachineConstantPool;
  // instead, it has its own constant pool implementation.
  // We create a new MachineConstantPool here to be compatible with the emitter.
  MachineConstantPool MCP;
  const hash_set<const Constant*> &pool = MF.getInfo()->getConstantPoolValues();
  for (hash_set<const Constant*>::const_iterator I = pool.begin(),
         E = pool.end();  I != E; ++I)
  {
    Constant *C = (Constant*)*I;
    unsigned idx = MCP.getConstantPoolIndex(C);
    DEBUG(std::cerr << "Constant[" << idx << "] = 0x" << (intptr_t)C << "\n");
    ConstantMap[C] = idx;
  }  
  MCE.emitConstantPool(&MCP);

  for (MachineFunction::iterator I = MF.begin(), E = MF.end(); I != E; ++I)
    emitBasicBlock(*I);
  MCE.finishFunction(MF);

  DEBUG(std::cerr << "Finishing fn " << MF.getFunction()->getName() << "\n");
  ConstantMap.clear();

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
        if (op.opLoBits32()) { loBits32=true; }
        if (op.opHiBits32()) { hiBits32=true; }
        if (op.opLoBits64()) { loBits64=true; }
        if (op.opHiBits64()) { hiBits64=true; }
        MI->SetMachineOperandConst(ii, MachineOperand::MO_SignExtendedImmed,
                                   branchTarget);
        if (loBits32) { MI->setOperandLo32(ii); }
        else if (hiBits32) { MI->setOperandHi32(ii); }
        else if (loBits64) { MI->setOperandLo64(ii); }
        else if (hiBits64) { MI->setOperandHi64(ii); }
        DEBUG(std::cerr << "Rewrote BB ref: ");
        unsigned fixedInstr = SparcV9CodeEmitter::getBinaryCodeForInstr(*MI);
        *Ref = fixedInstr;
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
    unsigned binCode = getBinaryCodeForInstr(**I);
    if (binCode == (1 << 30)) {
      // this is an invalid call: the addr is out of bounds. that means a code
      // sequence has already been emitted, and this is a no-op
      DEBUG(std::cerr << "Call supressed: already emitted far call.\n");
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

      } else if (Constant *C = ConstantPointerRef::get(V)) {
        if (ConstantMap.find(C) != ConstantMap.end()) {
          return (void*)
            (intptr_t)MCE.getConstantPoolEntryAddress(ConstantMap[C]);
        } else {
          std::cerr << "Constant: 0x" << std::hex << &*C << std::dec
                    << ", " << *V << " not found in ConstantMap!\n";
          abort();
        }
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
