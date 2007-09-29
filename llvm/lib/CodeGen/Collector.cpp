//===-- Collector.cpp - Garbage collection infrastructure -----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Gordon Henriksen and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements target- and collector-independent garbage collection
// infrastructure.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/Collector.h"
#include "llvm/IntrinsicInst.h"
#include "llvm/Module.h"
#include "llvm/PassManager.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/Target/TargetFrameInfo.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Support/Compiler.h"

using namespace llvm;

namespace {
  
  /// This pass rewrites calls to the llvm.gcread or llvm.gcwrite intrinsics,
  /// replacing them with simple loads and stores as directed by the Collector.
  /// This is useful for most garbage collectors.
  class VISIBILITY_HIDDEN LowerIntrinsics : public FunctionPass {
    const Collector &Coll;
    
    /// GCRootInt, GCReadInt, GCWriteInt - The function prototypes for the
    /// llvm.gc* intrinsics.
    Function *GCRootInt, *GCReadInt, *GCWriteInt;
    
    static bool CouldBecomeSafePoint(Instruction *I);
    static void InsertRootInitializers(Function &F,
                                       AllocaInst **Roots, unsigned Count);
    
  public:
    static char ID;
    
    LowerIntrinsics(const Collector &GC);
    const char *getPassName() const;
    
    bool doInitialization(Module &M);
    bool runOnFunction(Function &F);
  };
  
  
  /// This is a target-independent pass over the machine function representation
  /// to identify safe points for the garbage collector in the machine code. It 
  /// inserts labels at safe points and populates the GCInfo class.
  class VISIBILITY_HIDDEN MachineCodeAnalysis : public MachineFunctionPass {
    const Collector &Coll;
    const TargetMachine &Targ;
    
    CollectorMetadata *MD;
    MachineModuleInfo *MMI;
    const TargetInstrInfo *TII;
    MachineFrameInfo *MFI;
    
    void FindSafePoints(MachineFunction &MF);
    void VisitCallPoint(MachineBasicBlock::iterator MI);
    unsigned InsertLabel(MachineBasicBlock &MBB, 
                         MachineBasicBlock::iterator MI) const;
    
    void FindStackOffsets(MachineFunction &MF);
    
  public:
    static char ID;
    
    MachineCodeAnalysis(const Collector &C, const TargetMachine &T);
    const char *getPassName() const;
    void getAnalysisUsage(AnalysisUsage &AU) const;
    
    bool runOnMachineFunction(MachineFunction &MF);
  };
  
}

// -----------------------------------------------------------------------------

const Collector *llvm::TheCollector = 0;

Collector::Collector() :
  NeededSafePoints(0),
  CustomReadBarriers(false),
  CustomWriteBarriers(false),
  CustomRoots(false),
  InitRoots(true)
{}

Collector::~Collector() {}

void Collector::addLoweringPasses(FunctionPassManager &PM) const {
  if (NeedsDefaultLoweringPass())
    PM.add(new LowerIntrinsics(*this));

  if (NeedsCustomLoweringPass())
    PM.add(createCustomLoweringPass());
}

void Collector::addLoweringPasses(PassManager &PM) const {
  if (NeedsDefaultLoweringPass())
    PM.add(new LowerIntrinsics(*this));

  if (NeedsCustomLoweringPass())
    PM.add(createCustomLoweringPass());
}

void Collector::addGenericMachineCodePass(FunctionPassManager &PM,
                                          const TargetMachine &TM,
                                          bool Fast) const {
  if (needsSafePoints())
    PM.add(new MachineCodeAnalysis(*this, TM));
}

bool Collector::NeedsDefaultLoweringPass() const {
  // Default lowering is necessary only if read or write barriers have a default
  // action. The default for roots is no action.
  return !customWriteBarrier()
      || !customReadBarrier()
      || initializeRoots();
}

bool Collector::NeedsCustomLoweringPass() const {
  // Custom lowering is only necessary if enabled for some action.
  return customWriteBarrier()
      || customReadBarrier()
      || customRoots();
}

Pass *Collector::createCustomLoweringPass() const {
  cerr << "Collector must override createCustomLoweringPass.\n";
  abort();
  return 0;
}
    
void Collector::beginAssembly(Module &M, std::ostream &OS, AsmPrinter &AP,
                              const TargetAsmInfo &TAI) const {
  // Default is no action.
}
    
void Collector::finishAssembly(Module &M, CollectorModuleMetadata &CMM,
                               std::ostream &OS, AsmPrinter &AP,
                               const TargetAsmInfo &TAI) const {
  // Default is no action.
}

// -----------------------------------------------------------------------------

char LowerIntrinsics::ID = 0;

LowerIntrinsics::LowerIntrinsics(const Collector &C)
  : FunctionPass((intptr_t)&ID), Coll(C),
    GCRootInt(0), GCReadInt(0), GCWriteInt(0) {}

const char *LowerIntrinsics::getPassName() const {
  return "Lower Garbage Collection Instructions";
}
    
/// doInitialization - If this module uses the GC intrinsics, find them now. If
/// not, this pass does not do anything.
bool LowerIntrinsics::doInitialization(Module &M) {
  GCReadInt  = M.getFunction("llvm.gcread");
  GCWriteInt = M.getFunction("llvm.gcwrite");
  GCRootInt  = M.getFunction("llvm.gcroot");
  return false;
}

void LowerIntrinsics::InsertRootInitializers(Function &F, AllocaInst **Roots, 
                                                          unsigned Count) {
  // Scroll past alloca instructions.
  BasicBlock::iterator IP = F.getEntryBlock().begin();
  while (isa<AllocaInst>(IP)) ++IP;
  
  // Search for initializers in the initial BB.
  SmallPtrSet<AllocaInst*,16> InitedRoots;
  for (; !CouldBecomeSafePoint(IP); ++IP)
    if (StoreInst *SI = dyn_cast<StoreInst>(IP))
      if (AllocaInst *AI = dyn_cast<AllocaInst>(
            IntrinsicInst::StripPointerCasts(SI->getOperand(1))))
        InitedRoots.insert(AI);
  
  // Add root initializers.
  for (AllocaInst **I = Roots, **E = Roots + Count; I != E; ++I)
    if (!InitedRoots.count(*I))
      new StoreInst(ConstantPointerNull::get(cast<PointerType>(
                      cast<PointerType>((*I)->getType())->getElementType())),
                    *I, IP);
}

/// CouldBecomeSafePoint - Predicate to conservatively determine whether the
/// instruction could introduce a safe point.
bool LowerIntrinsics::CouldBecomeSafePoint(Instruction *I) {
  // The natural definition of instructions which could introduce safe points
  // are:
  // 
  //   - call, invoke (AfterCall, BeforeCall)
  //   - phis (Loops)
  //   - invoke, ret, unwind (Exit)
  // 
  // However, instructions as seemingly inoccuous as arithmetic can become
  // libcalls upon lowering (e.g., div i64 on a 32-bit platform), so instead
  // it is necessary to take a conservative approach.
  
  if (isa<AllocaInst>(I) || isa<GetElementPtrInst>(I) ||
      isa<StoreInst>(I) || isa<LoadInst>(I))
    return false;
  
  // llvm.gcroot is safe because it doesn't do anything at runtime.
  if (CallInst *CI = dyn_cast<CallInst>(I))
    if (Function *F = CI->getCalledFunction())
      if (unsigned IID = F->getIntrinsicID())
        if (IID == Intrinsic::gcroot)
          return false;
  
  return true;
}

/// runOnFunction - Replace gcread/gcwrite intrinsics with loads and stores.
/// Leave gcroot intrinsics; the code generator needs to see those.
bool LowerIntrinsics::runOnFunction(Function &F) {
  // Quick exit for programs that do not declare the intrinsics.
  if (!GCReadInt && !GCWriteInt && !GCRootInt) return false;
  
  bool LowerWr = !Coll.customWriteBarrier();
  bool LowerRd = !Coll.customReadBarrier();
  bool InitRoots = Coll.initializeRoots();
  
  SmallVector<AllocaInst*,32> Roots;
  
  bool MadeChange = false;
  for (Function::iterator BB = F.begin(), E = F.end(); BB != E; ++BB) {
    for (BasicBlock::iterator II = BB->begin(), E = BB->end(); II != E;) {
      if (CallInst *CI = dyn_cast<CallInst>(II++)) {
        Function *F = CI->getCalledFunction();
        if (F == GCWriteInt && LowerWr) {
          // Replace a write barrier with a simple store.
          Value *St = new StoreInst(CI->getOperand(1), CI->getOperand(3), CI);
          CI->replaceAllUsesWith(St);
          CI->eraseFromParent();
        } else if (F == GCReadInt && LowerRd) {
          // Replace a read barrier with a simple load.
          Value *Ld = new LoadInst(CI->getOperand(2), "", CI);
          Ld->takeName(CI);
          CI->replaceAllUsesWith(Ld);
          CI->eraseFromParent();
        } else if (F == GCRootInt && InitRoots) {
          // Initialize the GC root, but do not delete the intrinsic. The
          // backend needs the intrinsic to flag the stack slot.
          Roots.push_back(cast<AllocaInst>(
            IntrinsicInst::StripPointerCasts(CI->getOperand(1))));
        } else {
          continue;
        }
        
        MadeChange = true;
      }
    }
  }
  
  if (Roots.size())
    InsertRootInitializers(F, Roots.begin(), Roots.size());
  
  return MadeChange;
}

// -----------------------------------------------------------------------------

char MachineCodeAnalysis::ID = 0;

MachineCodeAnalysis::MachineCodeAnalysis(const Collector &C, const TargetMachine &T)
  : MachineFunctionPass(intptr_t(&ID)), Coll(C), Targ(T) {}

const char *MachineCodeAnalysis::getPassName() const {
  return "Analyze Machine Code For Garbage Collection";
}

void MachineCodeAnalysis::getAnalysisUsage(AnalysisUsage &AU) const {
  MachineFunctionPass::getAnalysisUsage(AU);
  AU.setPreservesAll();
  AU.addRequired<MachineModuleInfo>();
  AU.addRequired<CollectorModuleMetadata>();
}

unsigned MachineCodeAnalysis::InsertLabel(MachineBasicBlock &MBB, 
                                     MachineBasicBlock::iterator MI) const {
  unsigned Label = MMI->NextLabelID();
  BuildMI(MBB, MI, TII->get(TargetInstrInfo::LABEL)).addImm(Label);
  return Label;
}

void MachineCodeAnalysis::VisitCallPoint(MachineBasicBlock::iterator CI) {
  // Find the return address (next instruction), too, so as to bracket the call
  // instruction.
  MachineBasicBlock::iterator RAI = CI; 
  ++RAI;                                
  
  if (Coll.needsSafePoint(GC::PreCall))
    MD->addSafePoint(GC::PreCall, InsertLabel(*CI->getParent(), CI));
  
  if (Coll.needsSafePoint(GC::PostCall))
    MD->addSafePoint(GC::PostCall, InsertLabel(*CI->getParent(), RAI));
}

void MachineCodeAnalysis::FindSafePoints(MachineFunction &MF) {
  for (MachineFunction::iterator BBI = MF.begin(),
                                 BBE = MF.end(); BBI != BBE; ++BBI)
    for (MachineBasicBlock::iterator MI = BBI->begin(),
                                     ME = BBI->end(); MI != ME; ++MI)
      if (TII->isCall(MI->getOpcode()))
        VisitCallPoint(*MI);
}

void MachineCodeAnalysis::FindStackOffsets(MachineFunction &MF) {
  uint64_t StackSize = MFI->getStackSize();
  uint64_t OffsetAdjustment = MFI->getOffsetAdjustment();
  uint64_t OffsetOfLocalArea = Targ.getFrameInfo()->getOffsetOfLocalArea();
  
  for (CollectorMetadata::roots_iterator RI = MD->roots_begin(),
                                         RE = MD->roots_end(); RI != RE; ++RI)
    RI->StackOffset = MFI->getObjectOffset(RI->Num) + StackSize
                      - OffsetOfLocalArea + OffsetAdjustment;
}

bool MachineCodeAnalysis::runOnMachineFunction(MachineFunction &MF) {
  if (!Coll.needsSafePoints())
    return false;
  
  MD = getAnalysis<CollectorModuleMetadata>().get(MF.getFunction());
  MMI = &getAnalysis<MachineModuleInfo>();
  TII = MF.getTarget().getInstrInfo();
  MFI = MF.getFrameInfo();
  
  // Find the size of the stack frame.
  MD->setFrameSize(MFI->getStackSize());
  
  // Find all safe points.
  FindSafePoints(MF);
  
  // Find the stack offsets for all roots.
  FindStackOffsets(MF);
  
  return false;
}
