//===-- SparcV9CodeEmitter.cpp -  --------===//
//
//
//===----------------------------------------------------------------------===//

#include "llvm/Constants.h"
#include "llvm/Function.h"
#include "llvm/GlobalVariable.h"
#include "llvm/PassManager.h"
#include "llvm/CodeGen/MachineCodeEmitter.h"
#include "llvm/CodeGen/MachineFunctionInfo.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetData.h"
#include "Support/hash_set"
#include "SparcInternals.h"
#include "SparcV9CodeEmitter.h"

bool UltraSparc::addPassesToEmitMachineCode(PassManager &PM,
                                            MachineCodeEmitter &MCE) {
  //PM.add(new SparcV9CodeEmitter(MCE));
  //MachineCodeEmitter *M = MachineCodeEmitter::createDebugMachineCodeEmitter();
  MachineCodeEmitter *M = MachineCodeEmitter::createFilePrinterEmitter(MCE);
  PM.add(new SparcV9CodeEmitter(this, *M));
  PM.add(createMachineCodeDestructionPass()); // Free stuff no longer needed
  return false;
}

namespace {
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
  uint64_t *StackPtr = (uint64_t*)__builtin_frame_address(0);
  uint64_t RetAddr = (uint64_t)(intptr_t)__builtin_return_address(0);

#if 0  
  std::cerr << "In callback! Addr=0x" << std::hex << RetAddr
            << " SP=0x" << (unsigned)StackPtr << std::dec
            << ": Resolving call to function: "
            << TheVM->getFunctionReferencedName((void*)RetAddr) << "\n";
#endif

  std::cerr << "Sparc's JIT Resolver not implemented!\n";
  abort();

#if 0
  unsigned NewVal = TheJITResolver->resolveFunctionReference((void*)RetAddr);

  // Rewrite the call target... so that we don't fault every time we execute
  // the call.
  *(unsigned*)RetAddr = NewVal;
  
  // Change the return address to reexecute the call instruction...
  StackPtr[1] -= 4;
#endif
}

/// emitStubForFunction - This method is used by the JIT when it needs to emit
/// the address of a function for a function whose code has not yet been
/// generated.  In order to do this, it generates a stub which jumps to the lazy
/// function compiler, which will eventually get fixed to call the function
/// directly.
///
unsigned JITResolver::emitStubForFunction(Function *F) {
#if 0
  MCE.startFunctionStub(*F, 6);
  MCE.emitByte(0xE8);   // Call with 32 bit pc-rel destination...

  unsigned Address = addFunctionReference(MCE.getCurrentPCValue(), F);
  MCE.emitWord(Address-MCE.getCurrentPCValue()-4);

  MCE.emitByte(0xCD);   // Interrupt - Just a marker identifying the stub!
  return (intptr_t)MCE.finishFunctionStub(*F);
#endif
  std::cerr << "Sparc's JITResolver::emitStubForFunction() not implemented!\n";
  abort();
}


void SparcV9CodeEmitter::emitConstant(unsigned Val, unsigned Size) {
  // Output the constant in big endian byte order...
  unsigned byteVal;
  for (int i = Size-1; i >= 0; --i) {
    byteVal = Val >> 8*i;
    MCE->emitByte(byteVal & 255);
  }
}

unsigned getRealRegNum(unsigned fakeReg, unsigned regClass) {
  switch (regClass) {
  case UltraSparcRegInfo::IntRegType: {
    // Sparc manual, p31
    static const unsigned IntRegMap[] = {
      // "o0", "o1", "o2", "o3", "o4", "o5",       "o7",
      8, 9, 10, 11, 12, 13, 15,
      // "l0", "l1", "l2", "l3", "l4", "l5", "l6", "l7",
      16, 17, 18, 19, 20, 21, 22, 23,
      // "i0", "i1", "i2", "i3", "i4", "i5",  
      24, 25, 26, 27, 28, 29,
      // "i6", "i7",
      30, 31,
      // "g0", "g1", "g2", "g3", "g4", "g5",  "g6", "g7", 
      0, 1, 2, 3, 4, 5, 6, 7,
      // "o6"
      14
    }; 
 
    return IntRegMap[fakeReg];
    break;
  }
  case UltraSparcRegInfo::FPSingleRegType: {
    return fakeReg;
  }
  case UltraSparcRegInfo::FPDoubleRegType: {
    return fakeReg;
  }
  case UltraSparcRegInfo::FloatCCRegType: {
    return fakeReg;

  }
  case UltraSparcRegInfo::IntCCRegType: {
    return fakeReg;
  }
  default:
    assert(0 && "Invalid unified register number in getRegType");
    return fakeReg;
  }
}

int64_t SparcV9CodeEmitter::getMachineOpValue(MachineInstr &MI,
                                              MachineOperand &MO) {
  int64_t rv = 0; // Return value; defaults to 0 for unhandled cases
                  // or things that get fixed up later by the JIT.

  if (MO.isVirtualRegister()) {
    std::cerr << "ERROR: virtual register found in machine code.\n";
    abort();
  } else if (MO.isPCRelativeDisp()) {
    Value *V = MO.getVRegValue();
    if (BasicBlock *BB = dyn_cast<BasicBlock>(V)) {
      std::cerr << "Saving reference to BB (VReg)\n";
      unsigned* CurrPC = (unsigned*)(intptr_t)MCE->getCurrentPCValue();
      BBRefs.push_back(std::make_pair(BB, std::make_pair(CurrPC, &MI)));
    } else if (Constant *C = dyn_cast<Constant>(V)) {
      if (ConstantMap.find(C) != ConstantMap.end())
        rv = (int64_t)(intptr_t)ConstantMap[C];
      else {
        std::cerr << "ERROR: constant not in map:" << MO << "\n";
        abort();
      }
    } else {
      std::cerr << "ERROR: PC relative disp unhandled:" << MO << "\n";
      abort();
    }
  } else if (MO.isPhysicalRegister()) {
    // This is necessary because the Sparc doesn't actually lay out registers
    // in the real fashion -- it skips those that it chooses not to allocate,
    // i.e. those that are the SP, etc.
    unsigned fakeReg = MO.getReg(), realReg, regClass, regType;
    regType = TM->getRegInfo().getRegType(fakeReg);
    // At least map fakeReg into its class
    fakeReg = TM->getRegInfo().getClassRegNum(fakeReg, regClass);
    // Find the real register number for use in an instruction
    realReg = getRealRegNum(fakeReg, regClass);
    std::cerr << "Reg[" << std::dec << fakeReg << "] = " << realReg << "\n";
    rv = realReg;
  } else if (MO.isImmediate()) {
    rv = MO.getImmedValue();
  } else if (MO.isGlobalAddress()) {
    rv = (int64_t)
      (intptr_t)getGlobalAddress(cast<GlobalValue>(MO.getVRegValue()),
                                 MI, MO.isPCRelative());
  } else if (MO.isMachineBasicBlock()) {
    // Duplicate code of the above case for VirtualRegister, BasicBlock... 
    // It should really hit this case, but Sparc backend uses VRegs instead
    std::cerr << "Saving reference to MBB\n";
    BasicBlock *BB = MO.getMachineBasicBlock()->getBasicBlock();
    unsigned* CurrPC = (unsigned*)(intptr_t)MCE->getCurrentPCValue();
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
  if (MO.opLoBits32()) {          // %lo(val)
    return rv & 0x03ff;
  } else if (MO.opHiBits32()) {   // %lm(val)
    return (rv >> 10) & 0x03fffff;
  } else if (MO.opLoBits64()) {   // %hm(val)
    return (rv >> 32) & 0x03ff;
  } else if (MO.opHiBits64()) {   // %hh(val)
    return rv >> 42;
  } else {                        // (unadorned) val
    return rv;
  }
}

unsigned SparcV9CodeEmitter::getValueBit(int64_t Val, unsigned bit) {
  Val >>= bit;
  return (Val & 1);
}

void* SparcV9CodeEmitter::convertAddress(intptr_t Addr, bool isPCRelative) {
  if (isPCRelative) {
    return (void*)(Addr - (intptr_t)MCE->getCurrentPCValue());
  } else {
    return (void*)Addr;
  }
}



bool SparcV9CodeEmitter::runOnMachineFunction(MachineFunction &MF) {
  std::cerr << "Starting function " << MF.getFunction()->getName()
            << ", address: " << "0x" << std::hex 
            << (long)MCE->getCurrentPCValue() << "\n";

  MCE->startFunction(MF);

  // FIXME: the Sparc backend does not use the ConstantPool!!
  //MCE->emitConstantPool(MF.getConstantPool());

  // Instead, the Sparc backend has its own constant pool implementation:
  const hash_set<const Constant*> &pool = MF.getInfo()->getConstantPoolValues();
  for (hash_set<const Constant*>::const_iterator I = pool.begin(),
         E = pool.end();  I != E; ++I)
  {
    const Constant *C = *I;
    // For now we just allocate some memory on the heap, this can be
    // dramatically improved.
    const Type *Ty = ((Value*)C)->getType();
    void *Addr = malloc(TM->getTargetData().getTypeSize(Ty));
    //FIXME
    //TheVM.InitializeMemory(C, Addr);
    std::cerr << "Adding ConstantMap[" << C << "]=" << std::dec << Addr << "\n";
    ConstantMap[C] = Addr;
  }  

  for (MachineFunction::iterator I = MF.begin(), E = MF.end(); I != E; ++I)
    emitBasicBlock(*I);
  MCE->finishFunction(MF);

  std::cerr << "Finishing function " << MF.getFunction()->getName() << "\n";
  ConstantMap.clear();
  for (unsigned i = 0, e = BBRefs.size(); i != e; ++i) {
    long Location = BBLocations[BBRefs[i].first];
    unsigned *Ref = BBRefs[i].second.first;
    MachineInstr *MI = BBRefs[i].second.second;
    std::cerr << "Fixup @" << std::hex << Ref << " to " << Location
              << " in instr: " << std::dec << *MI << "\n";
  }

  // Resolve branches to BasicBlocks for the entire function
  for (unsigned i = 0, e = BBRefs.size(); i != e; ++i) {
    long Location = BBLocations[BBRefs[i].first];
    unsigned *Ref = BBRefs[i].second.first;
    MachineInstr *MI = BBRefs[i].second.second;
    std::cerr << "attempting to resolve BB: " << i << "\n";
    for (unsigned ii = 0, ee = MI->getNumOperands(); ii != ee; ++ii) {
      MachineOperand &op = MI->getOperand(ii);
      if (op.isPCRelativeDisp()) {
        // the instruction's branch target is made such that it branches to
        // PC + (br target * 4), so undo that arithmetic here:
        // Location is the target of the branch
        // Ref is the location of the instruction, and hence the PC
        unsigned branchTarget = (Location - (long)Ref) >> 2;
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
        std::cerr << "Rewrote BB ref: ";
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
  BBLocations[currBB] = MCE->getCurrentPCValue();
  for (MachineBasicBlock::iterator I = MBB.begin(), E = MBB.end(); I != E; ++I)
    emitInstruction(**I);
}

void SparcV9CodeEmitter::emitInstruction(MachineInstr &MI) {
  emitConstant(getBinaryCodeForInstr(MI), 4);
}

void* SparcV9CodeEmitter::getGlobalAddress(GlobalValue *V, MachineInstr &MI,
                                           bool isPCRelative)
{
  if (isPCRelative) { // must be a call, this is a major hack!
    // Try looking up the function to see if it is already compiled!
    if (void *Addr = (void*)(intptr_t)MCE->getGlobalValueAddress(V)) {
      intptr_t CurByte = MCE->getCurrentPCValue();
      // The real target of the call is Addr = PC + (target * 4)
      // CurByte is the PC, Addr we just received
      return (void*) (((long)Addr - (long)CurByte) >> 2);
    } else {
      if (Function *F = dyn_cast<Function>(V)) {
        // Function has not yet been code generated!
        TheJITResolver->addFunctionReference(MCE->getCurrentPCValue(),
                                             cast<Function>(V));
        // Delayed resolution...
        return 
          (void*)(intptr_t)TheJITResolver->getLazyResolver(cast<Function>(V));

      } else if (Constant *C = ConstantPointerRef::get(V)) {
        if (ConstantMap.find(C) != ConstantMap.end()) {
          return ConstantMap[C];
        } else {
          std::cerr << "Constant: 0x" << std::hex << &*C << std::dec
                    << ", " << *V << " not found in ConstantMap!\n";
          abort();
        }

#if 0
      } else if (const GlobalVariable *G = dyn_cast<GlobalVariable>(V)) {
        if (G->isConstant()) {
          const Constant* C = G->getInitializer();
          if (ConstantMap.find(C) != ConstantMap.end()) {
            return ConstantMap[C];
          } else {
            std::cerr << "Constant: " << *G << " not found in ConstantMap!\n";
            abort();
          }
        } else {
          std::cerr << "Variable: " << *G << " address not found!\n";
          abort();          
        }
#endif
      } else {
        std::cerr << "Unhandled global: " << *V << "\n";
        abort();
      }
    }
  } else {
    return convertAddress((intptr_t)MCE->getGlobalValueAddress(V),
                          isPCRelative);
  }
}


#include "SparcV9CodeEmitter.inc"

