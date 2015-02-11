//===----- X86CallFrameOptimization.cpp - Optimize x86 call sequences -----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines a pass that optimizes call sequences on x86.
// Currently, it converts movs of function parameters onto the stack into
// pushes. This is beneficial for two main reasons:
// 1) The push instruction encoding is much smaller than an esp-relative mov
// 2) It is possible to push memory arguments directly. So, if the
//    the transformation is preformed pre-reg-alloc, it can help relieve
//    register pressure.
//
//===----------------------------------------------------------------------===//

#include <algorithm>

#include "X86.h"
#include "X86InstrInfo.h"
#include "X86Subtarget.h"
#include "X86MachineFunctionInfo.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/IR/Function.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetInstrInfo.h"

using namespace llvm;

#define DEBUG_TYPE "x86-cf-opt"

static cl::opt<bool>
    NoX86CFOpt("no-x86-call-frame-opt",
               cl::desc("Avoid optimizing x86 call frames for size"),
               cl::init(false), cl::Hidden);

namespace {
class X86CallFrameOptimization : public MachineFunctionPass {
public:
  X86CallFrameOptimization() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override;

private:
  bool shouldPerformTransformation(MachineFunction &MF);

  // Information we know about a particular call site
  struct CallContext {
    CallContext()
        : Call(nullptr), SPCopy(nullptr), ExpectedDist(0),
          MovVector(4, nullptr), UsePush(false){};

    // Actuall call instruction
    MachineInstr *Call;

    // A copy of the stack pointer
    MachineInstr *SPCopy;

    // The total displacement of all passed parameters
    int64_t ExpectedDist;

    // The sequence of movs used to pass the parameters
    SmallVector<MachineInstr *, 4> MovVector;

    // Whether this site should use push instructions
    bool UsePush;
  };

  void collectCallInfo(MachineFunction &MF, MachineBasicBlock &MBB,
                       MachineBasicBlock::iterator I, CallContext &Context);

  bool adjustCallSequence(MachineFunction &MF, MachineBasicBlock::iterator I,
                          const CallContext &Context);

  MachineInstr *canFoldIntoRegPush(MachineBasicBlock::iterator FrameSetup,
                                   unsigned Reg);

  const char *getPassName() const override { return "X86 Optimize Call Frame"; }

  const TargetInstrInfo *TII;
  const TargetFrameLowering *TFL;
  const MachineRegisterInfo *MRI;
  static char ID;
};

char X86CallFrameOptimization::ID = 0;
}

FunctionPass *llvm::createX86CallFrameOptimization() {
  return new X86CallFrameOptimization();
}

// This checks whether the transformation is legal and profitable
bool X86CallFrameOptimization::shouldPerformTransformation(
    MachineFunction &MF) {
  if (NoX86CFOpt.getValue())
    return false;

  // We currently only support call sequences where *all* parameters.
  // are passed on the stack.
  // No point in running this in 64-bit mode, since some arguments are
  // passed in-register in all common calling conventions, so the pattern
  // we're looking for will never match.
  const X86Subtarget &STI = MF.getTarget().getSubtarget<X86Subtarget>();
  if (STI.is64Bit())
    return false;

  // You would expect straight-line code between call-frame setup and
  // call-frame destroy. You would be wrong. There are circumstances (e.g.
  // CMOV_GR8 expansion of a select that feeds a function call!) where we can
  // end up with the setup and the destroy in different basic blocks.
  // This is bad, and breaks SP adjustment.
  // So, check that all of the frames in the function are closed inside
  // the same block, and, for good measure, that there are no nested frames.
  int FrameSetupOpcode = TII->getCallFrameSetupOpcode();
  int FrameDestroyOpcode = TII->getCallFrameDestroyOpcode();
  for (MachineBasicBlock &BB : MF) {
    bool InsideFrameSequence = false;
    for (MachineInstr &MI : BB) {
      if (MI.getOpcode() == FrameSetupOpcode) {
        if (InsideFrameSequence)
          return false;
        InsideFrameSequence = true;
      } else if (MI.getOpcode() == FrameDestroyOpcode) {
        if (!InsideFrameSequence)
          return false;
        InsideFrameSequence = false;
      }
    }

    if (InsideFrameSequence)
      return false;
  }

  // Now that we know the transformation is legal, check if it is
  // profitable.
  // TODO: Add a heuristic that actually looks at the function,
  //       and enable this for more cases.

  // This transformation is always a win when we expected to have
  // a reserved call frame. Under other circumstances, it may be either
  // a win or a loss, and requires a heuristic.
  // For now, enable it only for the relatively clear win cases.
  bool CannotReserveFrame = MF.getFrameInfo()->hasVarSizedObjects();
  if (CannotReserveFrame)
    return true;

  // For now, don't even try to evaluate the profitability when
  // not optimizing for size.
  AttributeSet FnAttrs = MF.getFunction()->getAttributes();
  bool OptForSize =
      FnAttrs.hasAttribute(AttributeSet::FunctionIndex,
                           Attribute::OptimizeForSize) ||
      FnAttrs.hasAttribute(AttributeSet::FunctionIndex, Attribute::MinSize);

  if (!OptForSize)
    return false;

  // Stack re-alignment can make this unprofitable even in terms of size.
  // As mentioned above, a better heuristic is needed. For now, don't do this
  // when the required alignment is above 8. (4 would be the safe choice, but
  // some experimentation showed 8 is generally good).
  if (TFL->getStackAlignment() > 8)
    return false;

  return true;
}

bool X86CallFrameOptimization::runOnMachineFunction(MachineFunction &MF) {
  TII = MF.getSubtarget().getInstrInfo();
  TFL = MF.getSubtarget().getFrameLowering();
  MRI = &MF.getRegInfo();

  if (!shouldPerformTransformation(MF))
    return false;

  int FrameSetupOpcode = TII->getCallFrameSetupOpcode();

  bool Changed = false;

  DenseMap<MachineInstr *, CallContext> CallSeqMap;

  for (MachineFunction::iterator BB = MF.begin(), E = MF.end(); BB != E; ++BB)
    for (MachineBasicBlock::iterator I = BB->begin(); I != BB->end(); ++I)
      if (I->getOpcode() == FrameSetupOpcode) {
        CallContext &Context = CallSeqMap[I];
        collectCallInfo(MF, *BB, I, Context);
      }

  for (auto CC : CallSeqMap)
    if (CC.second.UsePush)
      Changed |= adjustCallSequence(MF, CC.first, CC.second);

  return Changed;
}

void X86CallFrameOptimization::collectCallInfo(MachineFunction &MF,
                                               MachineBasicBlock &MBB,
                                               MachineBasicBlock::iterator I,
                                               CallContext &Context) {
  // Check that this particular call sequence is amenable to the
  // transformation.
  const X86RegisterInfo &RegInfo = *static_cast<const X86RegisterInfo *>(
                                       MF.getSubtarget().getRegisterInfo());
  unsigned StackPtr = RegInfo.getStackRegister();
  int FrameDestroyOpcode = TII->getCallFrameDestroyOpcode();

  // We expect to enter this at the beginning of a call sequence
  assert(I->getOpcode() == TII->getCallFrameSetupOpcode());
  MachineBasicBlock::iterator FrameSetup = I++;

  // For globals in PIC mode, we can have some LEAs here.
  // Ignore them, they don't bother us.
  // TODO: Extend this to something that covers more cases.
  while (I->getOpcode() == X86::LEA32r)
    ++I;

  // We expect a copy instruction here.
  // TODO: The copy instruction is a lowering artifact.
  //       We should also support a copy-less version, where the stack
  //       pointer is used directly.
  if (!I->isCopy() || !I->getOperand(0).isReg())
    return;
  Context.SPCopy = I++;
  StackPtr = Context.SPCopy->getOperand(0).getReg();

  // Scan the call setup sequence for the pattern we're looking for.
  // We only handle a simple case - a sequence of MOV32mi or MOV32mr
  // instructions, that push a sequence of 32-bit values onto the stack, with
  // no gaps between them.
  unsigned int MaxAdjust = FrameSetup->getOperand(0).getImm() / 4;
  if (MaxAdjust > 4)
    Context.MovVector.resize(MaxAdjust, nullptr);

  do {
    int Opcode = I->getOpcode();
    if (Opcode != X86::MOV32mi && Opcode != X86::MOV32mr)
      break;

    // We only want movs of the form:
    // movl imm/r32, k(%esp)
    // If we run into something else, bail.
    // Note that AddrBaseReg may, counter to its name, not be a register,
    // but rather a frame index.
    // TODO: Support the fi case. This should probably work now that we
    // have the infrastructure to track the stack pointer within a call
    // sequence.
    if (!I->getOperand(X86::AddrBaseReg).isReg() ||
        (I->getOperand(X86::AddrBaseReg).getReg() != StackPtr) ||
        !I->getOperand(X86::AddrScaleAmt).isImm() ||
        (I->getOperand(X86::AddrScaleAmt).getImm() != 1) ||
        (I->getOperand(X86::AddrIndexReg).getReg() != X86::NoRegister) ||
        (I->getOperand(X86::AddrSegmentReg).getReg() != X86::NoRegister) ||
        !I->getOperand(X86::AddrDisp).isImm())
      return;

    int64_t StackDisp = I->getOperand(X86::AddrDisp).getImm();
    assert(StackDisp >= 0 &&
           "Negative stack displacement when passing parameters");

    // We really don't want to consider the unaligned case.
    if (StackDisp % 4)
      return;
    StackDisp /= 4;

    assert((size_t)StackDisp < Context.MovVector.size() &&
           "Function call has more parameters than the stack is adjusted for.");

    // If the same stack slot is being filled twice, something's fishy.
    if (Context.MovVector[StackDisp] != nullptr)
      return;
    Context.MovVector[StackDisp] = I;

    ++I;
  } while (I != MBB.end());

  // We now expect the end of the sequence - a call and a stack adjust.
  if (I == MBB.end())
    return;

  // For PCrel calls, we expect an additional COPY of the basereg.
  // If we find one, skip it.
  if (I->isCopy()) {
    if (I->getOperand(1).getReg() ==
        MF.getInfo<X86MachineFunctionInfo>()->getGlobalBaseReg())
      ++I;
    else
      return;
  }

  if (!I->isCall())
    return;

  Context.Call = I;
  if ((++I)->getOpcode() != FrameDestroyOpcode)
    return;

  // Now, go through the vector, and see that we don't have any gaps,
  // but only a series of 32-bit MOVs.
  auto MMI = Context.MovVector.begin(), MME = Context.MovVector.end();
  for (; MMI != MME; ++MMI, Context.ExpectedDist += 4)
    if (*MMI == nullptr)
      break;

  // If the call had no parameters, do nothing
  if (MMI == Context.MovVector.begin())
    return;

  // We are either at the last parameter, or a gap.
  // Make sure it's not a gap
  for (; MMI != MME; ++MMI)
    if (*MMI != nullptr)
      return;

  Context.UsePush = true;
  return;
}

bool X86CallFrameOptimization::adjustCallSequence(MachineFunction &MF,
                                                  MachineBasicBlock::iterator I,
                                                  const CallContext &Context) {
  // Ok, we can in fact do the transformation for this call.
  // Do not remove the FrameSetup instruction, but adjust the parameters.
  // PEI will end up finalizing the handling of this.
  MachineBasicBlock::iterator FrameSetup = I;
  MachineBasicBlock &MBB = *(I->getParent());
  FrameSetup->getOperand(1).setImm(Context.ExpectedDist);

  DebugLoc DL = I->getDebugLoc();
  // Now, iterate through the vector in reverse order, and replace the movs
  // with pushes. MOVmi/MOVmr doesn't have any defs, so no need to
  // replace uses.
  for (int Idx = (Context.ExpectedDist / 4) - 1; Idx >= 0; --Idx) {
    MachineBasicBlock::iterator MOV = *Context.MovVector[Idx];
    MachineOperand PushOp = MOV->getOperand(X86::AddrNumOperands);
    if (MOV->getOpcode() == X86::MOV32mi) {
      unsigned PushOpcode = X86::PUSHi32;
      // If the operand is a small (8-bit) immediate, we can use a
      // PUSH instruction with a shorter encoding.
      // Note that isImm() may fail even though this is a MOVmi, because
      // the operand can also be a symbol.
      if (PushOp.isImm()) {
        int64_t Val = PushOp.getImm();
        if (isInt<8>(Val))
          PushOpcode = X86::PUSH32i8;
      }
      BuildMI(MBB, Context.Call, DL, TII->get(PushOpcode)).addOperand(PushOp);
    } else {
      unsigned int Reg = PushOp.getReg();

      // If PUSHrmm is not slow on this target, try to fold the source of the
      // push into the instruction.
      const X86Subtarget &ST = MF.getTarget().getSubtarget<X86Subtarget>();
      bool SlowPUSHrmm = ST.isAtom() || ST.isSLM();

      // Check that this is legal to fold. Right now, we're extremely
      // conservative about that.
      MachineInstr *DefMov = nullptr;
      if (!SlowPUSHrmm && (DefMov = canFoldIntoRegPush(FrameSetup, Reg))) {
        MachineInstr *Push =
            BuildMI(MBB, Context.Call, DL, TII->get(X86::PUSH32rmm));

        unsigned NumOps = DefMov->getDesc().getNumOperands();
        for (unsigned i = NumOps - X86::AddrNumOperands; i != NumOps; ++i)
          Push->addOperand(DefMov->getOperand(i));

        DefMov->eraseFromParent();
      } else {
        BuildMI(MBB, Context.Call, DL, TII->get(X86::PUSH32r))
            .addReg(Reg)
            .getInstr();
      }
    }

    MBB.erase(MOV);
  }

  // The stack-pointer copy is no longer used in the call sequences.
  // There should not be any other users, but we can't commit to that, so:
  if (MRI->use_empty(Context.SPCopy->getOperand(0).getReg()))
    Context.SPCopy->eraseFromParent();

  // Once we've done this, we need to make sure PEI doesn't assume a reserved
  // frame.
  X86MachineFunctionInfo *FuncInfo = MF.getInfo<X86MachineFunctionInfo>();
  FuncInfo->setHasPushSequences(true);

  return true;
}

MachineInstr *X86CallFrameOptimization::canFoldIntoRegPush(
    MachineBasicBlock::iterator FrameSetup, unsigned Reg) {
  // Do an extremely restricted form of load folding.
  // ISel will often create patterns like:
  // movl    4(%edi), %eax
  // movl    8(%edi), %ecx
  // movl    12(%edi), %edx
  // movl    %edx, 8(%esp)
  // movl    %ecx, 4(%esp)
  // movl    %eax, (%esp)
  // call
  // Get rid of those with prejudice.
  if (!TargetRegisterInfo::isVirtualRegister(Reg))
    return nullptr;

  // Make sure this is the only use of Reg.
  if (!MRI->hasOneNonDBGUse(Reg))
    return nullptr;

  MachineBasicBlock::iterator DefMI = MRI->getVRegDef(Reg);

  // Make sure the def is a MOV from memory.
  // If the def is an another block, give up.
  if (DefMI->getOpcode() != X86::MOV32rm ||
      DefMI->getParent() != FrameSetup->getParent())
    return nullptr;

  // Be careful with movs that load from a stack slot, since it may get
  // resolved incorrectly.
  // TODO: Again, we already have the infrastructure, so this should work.
  if (!DefMI->getOperand(1).isReg())
    return nullptr;

  // Now, make sure everything else up until the ADJCALLSTACK is a sequence
  // of MOVs. To be less conservative would require duplicating a lot of the
  // logic from PeepholeOptimizer.
  // FIXME: A possibly better approach would be to teach the PeepholeOptimizer
  // to be smarter about folding into pushes.
  for (auto I = DefMI; I != FrameSetup; ++I)
    if (I->getOpcode() != X86::MOV32rm)
      return nullptr;

  return DefMI;
}
