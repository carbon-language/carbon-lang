//===-- X86/X86CodeEmitter.cpp - Convert X86 code to machine code ---------===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file contains the pass that transforms the X86 machine instructions into
// actual executable machine code.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "jit"
#include "X86TargetMachine.h"
#include "X86.h"
#include "llvm/PassManager.h"
#include "llvm/CodeGen/MachineCodeEmitter.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/Function.h"
#include "Support/Debug.h"
#include "Support/Statistic.h"
#include "Config/alloca.h"
using namespace llvm;

namespace {
  Statistic<>
  NumEmitted("x86-emitter", "Number of machine instructions emitted");

  class JITResolver {
    MachineCodeEmitter &MCE;

    // LazyCodeGenMap - Keep track of call sites for functions that are to be
    // lazily resolved.
    std::map<unsigned, Function*> LazyCodeGenMap;

    // LazyResolverMap - Keep track of the lazy resolver created for a
    // particular function so that we can reuse them if necessary.
    std::map<Function*, unsigned> LazyResolverMap;
  public:
    JITResolver(MachineCodeEmitter &mce) : MCE(mce) {}
    unsigned getLazyResolver(Function *F);
    unsigned addFunctionReference(unsigned Address, Function *F);
    
  private:
    unsigned emitStubForFunction(Function *F);
    static void CompilationCallback();
    unsigned resolveFunctionReference(unsigned RetAddr);
  };

  JITResolver *TheJITResolver;
}

void *X86TargetMachine::getJITStubForFunction(Function *F,
                                              MachineCodeEmitter &MCE) {
  if (TheJITResolver == 0)
    TheJITResolver = new JITResolver(MCE);
  return (void*)TheJITResolver->getLazyResolver(F);
}

/// addFunctionReference - This method is called when we need to emit the
/// address of a function that has not yet been emitted, so we don't know the
/// address.  Instead, we emit a call to the CompilationCallback method, and
/// keep track of where we are.
///
unsigned JITResolver::addFunctionReference(unsigned Address, Function *F) {
  LazyCodeGenMap[Address] = F;  
  return (intptr_t)&JITResolver::CompilationCallback;
}

unsigned JITResolver::resolveFunctionReference(unsigned RetAddr) {
  std::map<unsigned, Function*>::iterator I = LazyCodeGenMap.find(RetAddr);
  assert(I != LazyCodeGenMap.end() && "Not in map!");
  Function *F = I->second;
  LazyCodeGenMap.erase(I);
  return MCE.forceCompilationOf(F);
}

unsigned JITResolver::getLazyResolver(Function *F) {
  std::map<Function*, unsigned>::iterator I = LazyResolverMap.lower_bound(F);
  if (I != LazyResolverMap.end() && I->first == F) return I->second;
  
//std::cerr << "Getting lazy resolver for : " << ((Value*)F)->getName() << "\n";

  unsigned Stub = emitStubForFunction(F);
  LazyResolverMap.insert(I, std::make_pair(F, Stub));
  return Stub;
}

void JITResolver::CompilationCallback() {
  unsigned *StackPtr = (unsigned*)__builtin_frame_address(0);
  unsigned RetAddr = (unsigned)(intptr_t)__builtin_return_address(0);
  assert(StackPtr[1] == RetAddr &&
         "Could not find return address on the stack!");

  // It's a stub if there is an interrupt marker after the call...
  bool isStub = ((unsigned char*)(intptr_t)RetAddr)[0] == 0xCD;

  // FIXME FIXME FIXME FIXME: __builtin_frame_address doesn't work if frame
  // pointer elimination has been performed.  Having a variable sized alloca
  // disables frame pointer elimination currently, even if it's dead.  This is a
  // gross hack.
  alloca(10+isStub);
  // FIXME FIXME FIXME FIXME

  // The call instruction should have pushed the return value onto the stack...
  RetAddr -= 4;  // Backtrack to the reference itself...

#if 0
  DEBUG(std::cerr << "In callback! Addr=0x" << std::hex << RetAddr
                  << " ESP=0x" << (unsigned)StackPtr << std::dec
                  << ": Resolving call to function: "
                  << TheVM->getFunctionReferencedName((void*)RetAddr) << "\n");
#endif

  // Sanity check to make sure this really is a call instruction...
  assert(((unsigned char*)(intptr_t)RetAddr)[-1] == 0xE8 &&"Not a call instr!");
  
  unsigned NewVal = TheJITResolver->resolveFunctionReference(RetAddr);

  // Rewrite the call target... so that we don't fault every time we execute
  // the call.
  *(unsigned*)(intptr_t)RetAddr = NewVal-RetAddr-4;    

  if (isStub) {
    // If this is a stub, rewrite the call into an unconditional branch
    // instruction so that two return addresses are not pushed onto the stack
    // when the requested function finally gets called.  This also makes the
    // 0xCD byte (interrupt) dead, so the marker doesn't effect anything.
    ((unsigned char*)(intptr_t)RetAddr)[-1] = 0xE9;
  }

  // Change the return address to reexecute the call instruction...
  StackPtr[1] -= 5;
}

/// emitStubForFunction - This method is used by the JIT when it needs to emit
/// the address of a function for a function whose code has not yet been
/// generated.  In order to do this, it generates a stub which jumps to the lazy
/// function compiler, which will eventually get fixed to call the function
/// directly.
///
unsigned JITResolver::emitStubForFunction(Function *F) {
  MCE.startFunctionStub(*F, 6);
  MCE.emitByte(0xE8);   // Call with 32 bit pc-rel destination...

  unsigned Address = addFunctionReference(MCE.getCurrentPCValue(), F);
  MCE.emitWord(Address-MCE.getCurrentPCValue()-4);

  MCE.emitByte(0xCD);   // Interrupt - Just a marker identifying the stub!
  return (intptr_t)MCE.finishFunctionStub(*F);
}



namespace {
  class Emitter : public MachineFunctionPass {
    const X86InstrInfo  *II;
    MachineCodeEmitter  &MCE;
    std::map<const BasicBlock*, unsigned> BasicBlockAddrs;
    std::vector<std::pair<const BasicBlock*, unsigned> > BBRefs;
  public:
    Emitter(MachineCodeEmitter &mce) : II(0), MCE(mce) {}

    bool runOnMachineFunction(MachineFunction &MF);

    virtual const char *getPassName() const {
      return "X86 Machine Code Emitter";
    }

  private:
    void emitBasicBlock(MachineBasicBlock &MBB);
    void emitInstruction(MachineInstr &MI);

    void emitPCRelativeBlockAddress(BasicBlock *BB);
    void emitMaybePCRelativeValue(unsigned Address, bool isPCRelative);
    void emitGlobalAddressForCall(GlobalValue *GV);
    void emitGlobalAddressForPtr(GlobalValue *GV);

    void emitRegModRMByte(unsigned ModRMReg, unsigned RegOpcodeField);
    void emitSIBByte(unsigned SS, unsigned Index, unsigned Base);
    void emitConstant(unsigned Val, unsigned Size);

    void emitMemModRMByte(const MachineInstr &MI,
                          unsigned Op, unsigned RegOpcodeField);

  };
}

/// addPassesToEmitMachineCode - Add passes to the specified pass manager to get
/// machine code emitted.  This uses a MachineCodeEmitter object to handle
/// actually outputting the machine code and resolving things like the address
/// of functions.  This method should returns true if machine code emission is
/// not supported.
///
bool X86TargetMachine::addPassesToEmitMachineCode(FunctionPassManager &PM,
                                                  MachineCodeEmitter &MCE) {
  PM.add(new Emitter(MCE));
  return false;
}

bool Emitter::runOnMachineFunction(MachineFunction &MF) {
  II = &((X86TargetMachine&)MF.getTarget()).getInstrInfo();

  MCE.startFunction(MF);
  MCE.emitConstantPool(MF.getConstantPool());
  for (MachineFunction::iterator I = MF.begin(), E = MF.end(); I != E; ++I)
    emitBasicBlock(*I);
  MCE.finishFunction(MF);

  // Resolve all forward branches now...
  for (unsigned i = 0, e = BBRefs.size(); i != e; ++i) {
    unsigned Location = BasicBlockAddrs[BBRefs[i].first];
    unsigned Ref = BBRefs[i].second;
    *(unsigned*)(intptr_t)Ref = Location-Ref-4;
  }
  BBRefs.clear();
  BasicBlockAddrs.clear();
  return false;
}

void Emitter::emitBasicBlock(MachineBasicBlock &MBB) {
  if (uint64_t Addr = MCE.getCurrentPCValue())
    BasicBlockAddrs[MBB.getBasicBlock()] = Addr;

  for (MachineBasicBlock::iterator I = MBB.begin(), E = MBB.end(); I != E; ++I)
    emitInstruction(**I);
}


/// emitPCRelativeBlockAddress - This method emits the PC relative address of
/// the specified basic block, or if the basic block hasn't been emitted yet
/// (because this is a forward branch), it keeps track of the information
/// necessary to resolve this address later (and emits a dummy value).
///
void Emitter::emitPCRelativeBlockAddress(BasicBlock *BB) {
  // FIXME: Emit backward branches directly
  BBRefs.push_back(std::make_pair(BB, MCE.getCurrentPCValue()));
  MCE.emitWord(0);   // Emit a dummy value
}

/// emitMaybePCRelativeValue - Emit a 32-bit address which may be PC relative.
///
void Emitter::emitMaybePCRelativeValue(unsigned Address, bool isPCRelative) {
  if (isPCRelative)
    MCE.emitWord(Address-MCE.getCurrentPCValue()-4);
  else
    MCE.emitWord(Address);
}

/// emitGlobalAddressForCall - Emit the specified address to the code stream
/// assuming this is part of a function call, which is PC relative.
///
void Emitter::emitGlobalAddressForCall(GlobalValue *GV) {
  // Get the address from the backend...
  unsigned Address = MCE.getGlobalValueAddress(GV);
  
  if (Address == 0) {
    // FIXME: this is JIT specific!
    if (TheJITResolver == 0)
      TheJITResolver = new JITResolver(MCE);
    Address = TheJITResolver->addFunctionReference(MCE.getCurrentPCValue(),
                                                   cast<Function>(GV));
  }
  emitMaybePCRelativeValue(Address, true);
}

/// emitGlobalAddress - Emit the specified address to the code stream assuming
/// this is part of a "take the address of a global" instruction, which is not
/// PC relative.
///
void Emitter::emitGlobalAddressForPtr(GlobalValue *GV) {
  // Get the address from the backend...
  unsigned Address = MCE.getGlobalValueAddress(GV);

  // If the machine code emitter doesn't know what the address IS yet, we have
  // to take special measures.
  //
  if (Address == 0) {
    // FIXME: this is JIT specific!
    if (TheJITResolver == 0)
      TheJITResolver = new JITResolver(MCE);
    Address = TheJITResolver->getLazyResolver((Function*)GV);
  }

  emitMaybePCRelativeValue(Address, false);
}



/// N86 namespace - Native X86 Register numbers... used by X86 backend.
///
namespace N86 {
  enum {
    EAX = 0, ECX = 1, EDX = 2, EBX = 3, ESP = 4, EBP = 5, ESI = 6, EDI = 7
  };
}


// getX86RegNum - This function maps LLVM register identifiers to their X86
// specific numbering, which is used in various places encoding instructions.
//
static unsigned getX86RegNum(unsigned RegNo) {
  switch(RegNo) {
  case X86::EAX: case X86::AX: case X86::AL: return N86::EAX;
  case X86::ECX: case X86::CX: case X86::CL: return N86::ECX;
  case X86::EDX: case X86::DX: case X86::DL: return N86::EDX;
  case X86::EBX: case X86::BX: case X86::BL: return N86::EBX;
  case X86::ESP: case X86::SP: case X86::AH: return N86::ESP;
  case X86::EBP: case X86::BP: case X86::CH: return N86::EBP;
  case X86::ESI: case X86::SI: case X86::DH: return N86::ESI;
  case X86::EDI: case X86::DI: case X86::BH: return N86::EDI;

  case X86::ST0: case X86::ST1: case X86::ST2: case X86::ST3:
  case X86::ST4: case X86::ST5: case X86::ST6: case X86::ST7:
    return RegNo-X86::ST0;
  default:
    assert(RegNo >= MRegisterInfo::FirstVirtualRegister &&
           "Unknown physical register!");
    assert(0 && "Register allocator hasn't allocated reg correctly yet!");
    return 0;
  }
}

inline static unsigned char ModRMByte(unsigned Mod, unsigned RegOpcode,
                                      unsigned RM) {
  assert(Mod < 4 && RegOpcode < 8 && RM < 8 && "ModRM Fields out of range!");
  return RM | (RegOpcode << 3) | (Mod << 6);
}

void Emitter::emitRegModRMByte(unsigned ModRMReg, unsigned RegOpcodeFld){
  MCE.emitByte(ModRMByte(3, RegOpcodeFld, getX86RegNum(ModRMReg)));
}

void Emitter::emitSIBByte(unsigned SS, unsigned Index, unsigned Base) {
  // SIB byte is in the same format as the ModRMByte...
  MCE.emitByte(ModRMByte(SS, Index, Base));
}

void Emitter::emitConstant(unsigned Val, unsigned Size) {
  // Output the constant in little endian byte order...
  for (unsigned i = 0; i != Size; ++i) {
    MCE.emitByte(Val & 255);
    Val >>= 8;
  }
}

static bool isDisp8(int Value) {
  return Value == (signed char)Value;
}

void Emitter::emitMemModRMByte(const MachineInstr &MI,
                               unsigned Op, unsigned RegOpcodeField) {
  const MachineOperand &Disp     = MI.getOperand(Op+3);
  if (MI.getOperand(Op).isConstantPoolIndex()) {
    // Emit a direct address reference [disp32] where the displacement of the
    // constant pool entry is controlled by the MCE.
    MCE.emitByte(ModRMByte(0, RegOpcodeField, 5));
    unsigned Index = MI.getOperand(Op).getConstantPoolIndex();
    unsigned Address = MCE.getConstantPoolEntryAddress(Index);
    MCE.emitWord(Address+Disp.getImmedValue());
    return;
  }

  const MachineOperand &BaseReg  = MI.getOperand(Op);
  const MachineOperand &Scale    = MI.getOperand(Op+1);
  const MachineOperand &IndexReg = MI.getOperand(Op+2);

  // Is a SIB byte needed?
  if (IndexReg.getReg() == 0 && BaseReg.getReg() != X86::ESP) {
    if (BaseReg.getReg() == 0) {  // Just a displacement?
      // Emit special case [disp32] encoding
      MCE.emitByte(ModRMByte(0, RegOpcodeField, 5));
      emitConstant(Disp.getImmedValue(), 4);
    } else {
      unsigned BaseRegNo = getX86RegNum(BaseReg.getReg());
      if (Disp.getImmedValue() == 0 && BaseRegNo != N86::EBP) {
        // Emit simple indirect register encoding... [EAX] f.e.
        MCE.emitByte(ModRMByte(0, RegOpcodeField, BaseRegNo));
      } else if (isDisp8(Disp.getImmedValue())) {
        // Emit the disp8 encoding... [REG+disp8]
        MCE.emitByte(ModRMByte(1, RegOpcodeField, BaseRegNo));
        emitConstant(Disp.getImmedValue(), 1);
      } else {
        // Emit the most general non-SIB encoding: [REG+disp32]
        MCE.emitByte(ModRMByte(2, RegOpcodeField, BaseRegNo));
        emitConstant(Disp.getImmedValue(), 4);
      }
    }

  } else {  // We need a SIB byte, so start by outputting the ModR/M byte first
    assert(IndexReg.getReg() != X86::ESP && "Cannot use ESP as index reg!");

    bool ForceDisp32 = false;
    bool ForceDisp8  = false;
    if (BaseReg.getReg() == 0) {
      // If there is no base register, we emit the special case SIB byte with
      // MOD=0, BASE=5, to JUST get the index, scale, and displacement.
      MCE.emitByte(ModRMByte(0, RegOpcodeField, 4));
      ForceDisp32 = true;
    } else if (Disp.getImmedValue() == 0 && BaseReg.getReg() != X86::EBP) {
      // Emit no displacement ModR/M byte
      MCE.emitByte(ModRMByte(0, RegOpcodeField, 4));
    } else if (isDisp8(Disp.getImmedValue())) {
      // Emit the disp8 encoding...
      MCE.emitByte(ModRMByte(1, RegOpcodeField, 4));
      ForceDisp8 = true;           // Make sure to force 8 bit disp if Base=EBP
    } else {
      // Emit the normal disp32 encoding...
      MCE.emitByte(ModRMByte(2, RegOpcodeField, 4));
    }

    // Calculate what the SS field value should be...
    static const unsigned SSTable[] = { ~0, 0, 1, ~0, 2, ~0, ~0, ~0, 3 };
    unsigned SS = SSTable[Scale.getImmedValue()];

    if (BaseReg.getReg() == 0) {
      // Handle the SIB byte for the case where there is no base.  The
      // displacement has already been output.
      assert(IndexReg.getReg() && "Index register must be specified!");
      emitSIBByte(SS, getX86RegNum(IndexReg.getReg()), 5);
    } else {
      unsigned BaseRegNo = getX86RegNum(BaseReg.getReg());
      unsigned IndexRegNo;
      if (IndexReg.getReg())
	IndexRegNo = getX86RegNum(IndexReg.getReg());
      else
	IndexRegNo = 4;   // For example [ESP+1*<noreg>+4]
      emitSIBByte(SS, IndexRegNo, BaseRegNo);
    }

    // Do we need to output a displacement?
    if (Disp.getImmedValue() != 0 || ForceDisp32 || ForceDisp8) {
      if (!ForceDisp32 && isDisp8(Disp.getImmedValue()))
        emitConstant(Disp.getImmedValue(), 1);
      else
        emitConstant(Disp.getImmedValue(), 4);
    }
  }
}

static unsigned sizeOfPtr(const TargetInstrDescriptor &Desc) {
  switch (Desc.TSFlags & X86II::ArgMask) {
  case X86II::Arg8:   return 1;
  case X86II::Arg16:  return 2;
  case X86II::Arg32:  return 4;
  case X86II::ArgF32: return 4;
  case X86II::ArgF64: return 8;
  case X86II::ArgF80: return 10;
  default: assert(0 && "Memory size not set!");
    return 0;
  }
}

void Emitter::emitInstruction(MachineInstr &MI) {
  NumEmitted++;  // Keep track of the # of mi's emitted

  unsigned Opcode = MI.getOpcode();
  const TargetInstrDescriptor &Desc = II->get(Opcode);

  // Emit instruction prefixes if necessary
  if (Desc.TSFlags & X86II::OpSize) MCE.emitByte(0x66);// Operand size...

  switch (Desc.TSFlags & X86II::Op0Mask) {
  case X86II::TB:
    MCE.emitByte(0x0F);   // Two-byte opcode prefix
    break;
  case X86II::D8: case X86II::D9: case X86II::DA: case X86II::DB:
  case X86II::DC: case X86II::DD: case X86II::DE: case X86II::DF:
    MCE.emitByte(0xD8+
		 (((Desc.TSFlags & X86II::Op0Mask)-X86II::D8)
		                   >> X86II::Op0Shift));
    break; // Two-byte opcode prefix
  default: assert(0 && "Invalid prefix!");
  case 0: break;  // No prefix!
  }

  unsigned char BaseOpcode = II->getBaseOpcodeFor(Opcode);
  switch (Desc.TSFlags & X86II::FormMask) {
  default: assert(0 && "Unknown FormMask value in X86 MachineCodeEmitter!");
  case X86II::Pseudo:
    if (Opcode != X86::IMPLICIT_USE && Opcode != X86::IMPLICIT_DEF)
      std::cerr << "X86 Machine Code Emitter: No 'form', not emitting: " << MI;
    break;

  case X86II::RawFrm:
    MCE.emitByte(BaseOpcode);
    if (MI.getNumOperands() == 1) {
      MachineOperand &MO = MI.getOperand(0);
      if (MO.isPCRelativeDisp()) {
        // Conditional branch... FIXME: this should use an MBB destination!
        emitPCRelativeBlockAddress(cast<BasicBlock>(MO.getVRegValue()));
      } else if (MO.isGlobalAddress()) {
        assert(MO.isPCRelative() && "Call target is not PC Relative?");
        emitGlobalAddressForCall(MO.getGlobal());
      } else if (MO.isExternalSymbol()) {
        unsigned Address = MCE.getGlobalValueAddress(MO.getSymbolName());
        assert(Address && "Unknown external symbol!");
        emitMaybePCRelativeValue(Address, MO.isPCRelative());
      } else {
	assert(0 && "Unknown RawFrm operand!");
      }
    }
    break;

  case X86II::AddRegFrm:
    MCE.emitByte(BaseOpcode + getX86RegNum(MI.getOperand(0).getReg()));
    if (MI.getNumOperands() == 2) {
      MachineOperand &MO1 = MI.getOperand(1);
      if (MO1.isImmediate() || MO1.getVRegValueOrNull() ||
	  MO1.isGlobalAddress() || MO1.isExternalSymbol()) {
	unsigned Size = sizeOfPtr(Desc);
	if (Value *V = MO1.getVRegValueOrNull()) {
	  assert(Size == 4 && "Don't know how to emit non-pointer values!");
          emitGlobalAddressForPtr(cast<GlobalValue>(V));
	} else if (MO1.isGlobalAddress()) {
	  assert(Size == 4 && "Don't know how to emit non-pointer values!");
          assert(!MO1.isPCRelative() && "Function pointer ref is PC relative?");
          emitGlobalAddressForPtr(MO1.getGlobal());
	} else if (MO1.isExternalSymbol()) {
	  assert(Size == 4 && "Don't know how to emit non-pointer values!");

          unsigned Address = MCE.getGlobalValueAddress(MO1.getSymbolName());
          assert(Address && "Unknown external symbol!");
          emitMaybePCRelativeValue(Address, MO1.isPCRelative());
	} else {
	  emitConstant(MO1.getImmedValue(), Size);
	}
      }
    }
    break;

  case X86II::MRMDestReg: {
    MCE.emitByte(BaseOpcode);
    MachineOperand &SrcOp = MI.getOperand(1+II->isTwoAddrInstr(Opcode));
    emitRegModRMByte(MI.getOperand(0).getReg(), getX86RegNum(SrcOp.getReg()));
    if (MI.getNumOperands() == 4)
      emitConstant(MI.getOperand(3).getImmedValue(), sizeOfPtr(Desc));
    break;
  }
  case X86II::MRMDestMem:
    MCE.emitByte(BaseOpcode);
    emitMemModRMByte(MI, 0, getX86RegNum(MI.getOperand(4).getReg()));
    break;

  case X86II::MRMSrcReg:
    MCE.emitByte(BaseOpcode);

    if (MI.getNumOperands() == 2) {
      emitRegModRMByte(MI.getOperand(MI.getNumOperands()-1).getReg(),
                       getX86RegNum(MI.getOperand(0).getReg()));
    } else if (MI.getOperand(2).isImmediate()) {
      emitRegModRMByte(MI.getOperand(1).getReg(),
                       getX86RegNum(MI.getOperand(0).getReg()));

      emitConstant(MI.getOperand(2).getImmedValue(), sizeOfPtr(Desc));
    } else {
      emitRegModRMByte(MI.getOperand(2).getReg(),
                       getX86RegNum(MI.getOperand(0).getReg()));
    }
    break;

  case X86II::MRMSrcMem:
    MCE.emitByte(BaseOpcode);
    emitMemModRMByte(MI, MI.getNumOperands()-4,
                     getX86RegNum(MI.getOperand(0).getReg()));
    break;

  case X86II::MRMS0r: case X86II::MRMS1r:
  case X86II::MRMS2r: case X86II::MRMS3r:
  case X86II::MRMS4r: case X86II::MRMS5r:
  case X86II::MRMS6r: case X86II::MRMS7r:
    MCE.emitByte(BaseOpcode);
    emitRegModRMByte(MI.getOperand(0).getReg(),
                     (Desc.TSFlags & X86II::FormMask)-X86II::MRMS0r);

    if (MI.getOperand(MI.getNumOperands()-1).isImmediate()) {
      unsigned Size = sizeOfPtr(Desc);
      emitConstant(MI.getOperand(MI.getNumOperands()-1).getImmedValue(), Size);
    }
    break;

  case X86II::MRMS0m: case X86II::MRMS1m:
  case X86II::MRMS2m: case X86II::MRMS3m:
  case X86II::MRMS4m: case X86II::MRMS5m:
  case X86II::MRMS6m: case X86II::MRMS7m: 
    MCE.emitByte(BaseOpcode);
    emitMemModRMByte(MI, 0, (Desc.TSFlags & X86II::FormMask)-X86II::MRMS0m);

    if (MI.getNumOperands() == 5) {
      unsigned Size = sizeOfPtr(Desc);
      emitConstant(MI.getOperand(4).getImmedValue(), Size);
    }
    break;
  }
}
