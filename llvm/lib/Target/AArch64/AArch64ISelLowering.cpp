//===-- AArch64ISelLowering.cpp - AArch64 DAG Lowering Implementation -----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the interfaces that AArch64 uses to lower LLVM code into a
// selection DAG.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "aarch64-isel"
#include "AArch64.h"
#include "AArch64ISelLowering.h"
#include "AArch64MachineFunctionInfo.h"
#include "AArch64TargetMachine.h"
#include "AArch64TargetObjectFile.h"
#include "Utils/AArch64BaseInfo.h"
#include "llvm/CodeGen/Analysis.h"
#include "llvm/CodeGen/CallingConvLower.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/TargetLoweringObjectFileImpl.h"
#include "llvm/IR/CallingConv.h"

using namespace llvm;

static TargetLoweringObjectFile *createTLOF(AArch64TargetMachine &TM) {
  const AArch64Subtarget *Subtarget = &TM.getSubtarget<AArch64Subtarget>();

  if (Subtarget->isTargetLinux())
    return new AArch64LinuxTargetObjectFile();
  if (Subtarget->isTargetELF())
    return new TargetLoweringObjectFileELF();
  llvm_unreachable("unknown subtarget type");
}

AArch64TargetLowering::AArch64TargetLowering(AArch64TargetMachine &TM)
  : TargetLowering(TM, createTLOF(TM)), Itins(TM.getInstrItineraryData()) {

  // SIMD compares set the entire lane's bits to 1
  setBooleanVectorContents(ZeroOrNegativeOneBooleanContent);

  // Scalar register <-> type mapping
  addRegisterClass(MVT::i32, &AArch64::GPR32RegClass);
  addRegisterClass(MVT::i64, &AArch64::GPR64RegClass);
  addRegisterClass(MVT::f16, &AArch64::FPR16RegClass);
  addRegisterClass(MVT::f32, &AArch64::FPR32RegClass);
  addRegisterClass(MVT::f64, &AArch64::FPR64RegClass);
  addRegisterClass(MVT::f128, &AArch64::FPR128RegClass);

  computeRegisterProperties();

  // We combine OR nodes for bitfield and NEON BSL operations.
  setTargetDAGCombine(ISD::OR);

  setTargetDAGCombine(ISD::AND);
  setTargetDAGCombine(ISD::SRA);

  // AArch64 does not have i1 loads, or much of anything for i1 really.
  setLoadExtAction(ISD::SEXTLOAD, MVT::i1, Promote);
  setLoadExtAction(ISD::ZEXTLOAD, MVT::i1, Promote);
  setLoadExtAction(ISD::EXTLOAD, MVT::i1, Promote);

  setStackPointerRegisterToSaveRestore(AArch64::XSP);
  setOperationAction(ISD::DYNAMIC_STACKALLOC, MVT::i64, Expand);
  setOperationAction(ISD::STACKRESTORE, MVT::Other, Expand);
  setOperationAction(ISD::STACKSAVE, MVT::Other, Expand);

  // We'll lower globals to wrappers for selection.
  setOperationAction(ISD::GlobalAddress, MVT::i64, Custom);
  setOperationAction(ISD::GlobalTLSAddress, MVT::i64, Custom);

  // A64 instructions have the comparison predicate attached to the user of the
  // result, but having a separate comparison is valuable for matching.
  setOperationAction(ISD::BR_CC, MVT::i32, Custom);
  setOperationAction(ISD::BR_CC, MVT::i64, Custom);
  setOperationAction(ISD::BR_CC, MVT::f32, Custom);
  setOperationAction(ISD::BR_CC, MVT::f64, Custom);

  setOperationAction(ISD::SELECT, MVT::i32, Custom);
  setOperationAction(ISD::SELECT, MVT::i64, Custom);
  setOperationAction(ISD::SELECT, MVT::f32, Custom);
  setOperationAction(ISD::SELECT, MVT::f64, Custom);

  setOperationAction(ISD::SELECT_CC, MVT::i32, Custom);
  setOperationAction(ISD::SELECT_CC, MVT::i64, Custom);
  setOperationAction(ISD::SELECT_CC, MVT::f32, Custom);
  setOperationAction(ISD::SELECT_CC, MVT::f64, Custom);

  setOperationAction(ISD::BRCOND, MVT::Other, Custom);

  setOperationAction(ISD::SETCC, MVT::i32, Custom);
  setOperationAction(ISD::SETCC, MVT::i64, Custom);
  setOperationAction(ISD::SETCC, MVT::f32, Custom);
  setOperationAction(ISD::SETCC, MVT::f64, Custom);

  setOperationAction(ISD::BR_JT, MVT::Other, Expand);
  setOperationAction(ISD::JumpTable, MVT::i32, Custom);
  setOperationAction(ISD::JumpTable, MVT::i64, Custom);

  setOperationAction(ISD::VASTART, MVT::Other, Custom);
  setOperationAction(ISD::VACOPY, MVT::Other, Custom);
  setOperationAction(ISD::VAEND, MVT::Other, Expand);
  setOperationAction(ISD::VAARG, MVT::Other, Expand);

  setOperationAction(ISD::BlockAddress, MVT::i64, Custom);

  setOperationAction(ISD::ROTL, MVT::i32, Expand);
  setOperationAction(ISD::ROTL, MVT::i64, Expand);

  setOperationAction(ISD::UREM, MVT::i32, Expand);
  setOperationAction(ISD::UREM, MVT::i64, Expand);
  setOperationAction(ISD::UDIVREM, MVT::i32, Expand);
  setOperationAction(ISD::UDIVREM, MVT::i64, Expand);

  setOperationAction(ISD::SREM, MVT::i32, Expand);
  setOperationAction(ISD::SREM, MVT::i64, Expand);
  setOperationAction(ISD::SDIVREM, MVT::i32, Expand);
  setOperationAction(ISD::SDIVREM, MVT::i64, Expand);

  setOperationAction(ISD::CTPOP, MVT::i32, Expand);
  setOperationAction(ISD::CTPOP, MVT::i64, Expand);

  // Legal floating-point operations.
  setOperationAction(ISD::FABS, MVT::f32, Legal);
  setOperationAction(ISD::FABS, MVT::f64, Legal);

  setOperationAction(ISD::FCEIL, MVT::f32, Legal);
  setOperationAction(ISD::FCEIL, MVT::f64, Legal);

  setOperationAction(ISD::FFLOOR, MVT::f32, Legal);
  setOperationAction(ISD::FFLOOR, MVT::f64, Legal);

  setOperationAction(ISD::FNEARBYINT, MVT::f32, Legal);
  setOperationAction(ISD::FNEARBYINT, MVT::f64, Legal);

  setOperationAction(ISD::FNEG, MVT::f32, Legal);
  setOperationAction(ISD::FNEG, MVT::f64, Legal);

  setOperationAction(ISD::FRINT, MVT::f32, Legal);
  setOperationAction(ISD::FRINT, MVT::f64, Legal);

  setOperationAction(ISD::FSQRT, MVT::f32, Legal);
  setOperationAction(ISD::FSQRT, MVT::f64, Legal);

  setOperationAction(ISD::FTRUNC, MVT::f32, Legal);
  setOperationAction(ISD::FTRUNC, MVT::f64, Legal);

  setOperationAction(ISD::ConstantFP, MVT::f32, Legal);
  setOperationAction(ISD::ConstantFP, MVT::f64, Legal);
  setOperationAction(ISD::ConstantFP, MVT::f128, Legal);

  // Illegal floating-point operations.
  setOperationAction(ISD::FCOPYSIGN, MVT::f32, Expand);
  setOperationAction(ISD::FCOPYSIGN, MVT::f64, Expand);

  setOperationAction(ISD::FCOS, MVT::f32, Expand);
  setOperationAction(ISD::FCOS, MVT::f64, Expand);

  setOperationAction(ISD::FEXP, MVT::f32, Expand);
  setOperationAction(ISD::FEXP, MVT::f64, Expand);

  setOperationAction(ISD::FEXP2, MVT::f32, Expand);
  setOperationAction(ISD::FEXP2, MVT::f64, Expand);

  setOperationAction(ISD::FLOG, MVT::f32, Expand);
  setOperationAction(ISD::FLOG, MVT::f64, Expand);

  setOperationAction(ISD::FLOG2, MVT::f32, Expand);
  setOperationAction(ISD::FLOG2, MVT::f64, Expand);

  setOperationAction(ISD::FLOG10, MVT::f32, Expand);
  setOperationAction(ISD::FLOG10, MVT::f64, Expand);

  setOperationAction(ISD::FPOW, MVT::f32, Expand);
  setOperationAction(ISD::FPOW, MVT::f64, Expand);

  setOperationAction(ISD::FPOWI, MVT::f32, Expand);
  setOperationAction(ISD::FPOWI, MVT::f64, Expand);

  setOperationAction(ISD::FREM, MVT::f32, Expand);
  setOperationAction(ISD::FREM, MVT::f64, Expand);

  setOperationAction(ISD::FSIN, MVT::f32, Expand);
  setOperationAction(ISD::FSIN, MVT::f64, Expand);

  setOperationAction(ISD::FSINCOS, MVT::f32, Expand);
  setOperationAction(ISD::FSINCOS, MVT::f64, Expand);

  // Virtually no operation on f128 is legal, but LLVM can't expand them when
  // there's a valid register class, so we need custom operations in most cases.
  setOperationAction(ISD::FABS,       MVT::f128, Expand);
  setOperationAction(ISD::FADD,       MVT::f128, Custom);
  setOperationAction(ISD::FCOPYSIGN,  MVT::f128, Expand);
  setOperationAction(ISD::FCOS,       MVT::f128, Expand);
  setOperationAction(ISD::FDIV,       MVT::f128, Custom);
  setOperationAction(ISD::FMA,        MVT::f128, Expand);
  setOperationAction(ISD::FMUL,       MVT::f128, Custom);
  setOperationAction(ISD::FNEG,       MVT::f128, Expand);
  setOperationAction(ISD::FP_EXTEND,  MVT::f128, Expand);
  setOperationAction(ISD::FP_ROUND,   MVT::f128, Expand);
  setOperationAction(ISD::FPOW,       MVT::f128, Expand);
  setOperationAction(ISD::FREM,       MVT::f128, Expand);
  setOperationAction(ISD::FRINT,      MVT::f128, Expand);
  setOperationAction(ISD::FSIN,       MVT::f128, Expand);
  setOperationAction(ISD::FSINCOS,    MVT::f128, Expand);
  setOperationAction(ISD::FSQRT,      MVT::f128, Expand);
  setOperationAction(ISD::FSUB,       MVT::f128, Custom);
  setOperationAction(ISD::FTRUNC,     MVT::f128, Expand);
  setOperationAction(ISD::SETCC,      MVT::f128, Custom);
  setOperationAction(ISD::BR_CC,      MVT::f128, Custom);
  setOperationAction(ISD::SELECT,     MVT::f128, Expand);
  setOperationAction(ISD::SELECT_CC,  MVT::f128, Custom);
  setOperationAction(ISD::FP_EXTEND,  MVT::f128, Custom);

  // Lowering for many of the conversions is actually specified by the non-f128
  // type. The LowerXXX function will be trivial when f128 isn't involved.
  setOperationAction(ISD::FP_TO_SINT, MVT::i32, Custom);
  setOperationAction(ISD::FP_TO_SINT, MVT::i64, Custom);
  setOperationAction(ISD::FP_TO_SINT, MVT::i128, Custom);
  setOperationAction(ISD::FP_TO_UINT, MVT::i32, Custom);
  setOperationAction(ISD::FP_TO_UINT, MVT::i64, Custom);
  setOperationAction(ISD::FP_TO_UINT, MVT::i128, Custom);
  setOperationAction(ISD::SINT_TO_FP, MVT::i32, Custom);
  setOperationAction(ISD::SINT_TO_FP, MVT::i64, Custom);
  setOperationAction(ISD::SINT_TO_FP, MVT::i128, Custom);
  setOperationAction(ISD::UINT_TO_FP, MVT::i32, Custom);
  setOperationAction(ISD::UINT_TO_FP, MVT::i64, Custom);
  setOperationAction(ISD::UINT_TO_FP, MVT::i128, Custom);
  setOperationAction(ISD::FP_ROUND,  MVT::f32, Custom);
  setOperationAction(ISD::FP_ROUND,  MVT::f64, Custom);

  // This prevents LLVM trying to compress double constants into a floating
  // constant-pool entry and trying to load from there. It's of doubtful benefit
  // for A64: we'd need LDR followed by FCVT, I believe.
  setLoadExtAction(ISD::EXTLOAD, MVT::f64, Expand);
  setLoadExtAction(ISD::EXTLOAD, MVT::f32, Expand);
  setLoadExtAction(ISD::EXTLOAD, MVT::f16, Expand);

  setTruncStoreAction(MVT::f128, MVT::f64, Expand);
  setTruncStoreAction(MVT::f128, MVT::f32, Expand);
  setTruncStoreAction(MVT::f128, MVT::f16, Expand);
  setTruncStoreAction(MVT::f64, MVT::f32, Expand);
  setTruncStoreAction(MVT::f64, MVT::f16, Expand);
  setTruncStoreAction(MVT::f32, MVT::f16, Expand);

  setExceptionPointerRegister(AArch64::X0);
  setExceptionSelectorRegister(AArch64::X1);
}

EVT AArch64TargetLowering::getSetCCResultType(LLVMContext &, EVT VT) const {
  // It's reasonably important that this value matches the "natural" legal
  // promotion from i1 for scalar types. Otherwise LegalizeTypes can get itself
  // in a twist (e.g. inserting an any_extend which then becomes i64 -> i64).
  if (!VT.isVector()) return MVT::i32;
  return VT.changeVectorElementTypeToInteger();
}

static void getExclusiveOperation(unsigned Size, AtomicOrdering Ord,
                                  unsigned &LdrOpc,
                                  unsigned &StrOpc) {
  static unsigned LoadBares[] = {AArch64::LDXR_byte, AArch64::LDXR_hword,
                                 AArch64::LDXR_word, AArch64::LDXR_dword};
  static unsigned LoadAcqs[] = {AArch64::LDAXR_byte, AArch64::LDAXR_hword,
                                AArch64::LDAXR_word, AArch64::LDAXR_dword};
  static unsigned StoreBares[] = {AArch64::STXR_byte, AArch64::STXR_hword,
                                  AArch64::STXR_word, AArch64::STXR_dword};
  static unsigned StoreRels[] = {AArch64::STLXR_byte, AArch64::STLXR_hword,
                                 AArch64::STLXR_word, AArch64::STLXR_dword};

  unsigned *LoadOps, *StoreOps;
  if (Ord == Acquire || Ord == AcquireRelease || Ord == SequentiallyConsistent)
    LoadOps = LoadAcqs;
  else
    LoadOps = LoadBares;

  if (Ord == Release || Ord == AcquireRelease || Ord == SequentiallyConsistent)
    StoreOps = StoreRels;
  else
    StoreOps = StoreBares;

  assert(isPowerOf2_32(Size) && Size <= 8 &&
         "unsupported size for atomic binary op!");

  LdrOpc = LoadOps[Log2_32(Size)];
  StrOpc = StoreOps[Log2_32(Size)];
}

MachineBasicBlock *
AArch64TargetLowering::emitAtomicBinary(MachineInstr *MI, MachineBasicBlock *BB,
                                        unsigned Size,
                                        unsigned BinOpcode) const {
  // This also handles ATOMIC_SWAP, indicated by BinOpcode==0.
  const TargetInstrInfo *TII = getTargetMachine().getInstrInfo();

  const BasicBlock *LLVM_BB = BB->getBasicBlock();
  MachineFunction *MF = BB->getParent();
  MachineFunction::iterator It = BB;
  ++It;

  unsigned dest = MI->getOperand(0).getReg();
  unsigned ptr = MI->getOperand(1).getReg();
  unsigned incr = MI->getOperand(2).getReg();
  AtomicOrdering Ord = static_cast<AtomicOrdering>(MI->getOperand(3).getImm());
  DebugLoc dl = MI->getDebugLoc();

  MachineRegisterInfo &MRI = BB->getParent()->getRegInfo();

  unsigned ldrOpc, strOpc;
  getExclusiveOperation(Size, Ord, ldrOpc, strOpc);

  MachineBasicBlock *loopMBB = MF->CreateMachineBasicBlock(LLVM_BB);
  MachineBasicBlock *exitMBB = MF->CreateMachineBasicBlock(LLVM_BB);
  MF->insert(It, loopMBB);
  MF->insert(It, exitMBB);

  // Transfer the remainder of BB and its successor edges to exitMBB.
  exitMBB->splice(exitMBB->begin(), BB,
                  llvm::next(MachineBasicBlock::iterator(MI)),
                  BB->end());
  exitMBB->transferSuccessorsAndUpdatePHIs(BB);

  const TargetRegisterClass *TRC
    = Size == 8 ? &AArch64::GPR64RegClass : &AArch64::GPR32RegClass;
  unsigned scratch = (!BinOpcode) ? incr : MRI.createVirtualRegister(TRC);

  //  thisMBB:
  //   ...
  //   fallthrough --> loopMBB
  BB->addSuccessor(loopMBB);

  //  loopMBB:
  //   ldxr dest, ptr
  //   <binop> scratch, dest, incr
  //   stxr stxr_status, scratch, ptr
  //   cbnz stxr_status, loopMBB
  //   fallthrough --> exitMBB
  BB = loopMBB;
  BuildMI(BB, dl, TII->get(ldrOpc), dest).addReg(ptr);
  if (BinOpcode) {
    // All arithmetic operations we'll be creating are designed to take an extra
    // shift or extend operand, which we can conveniently set to zero.

    // Operand order needs to go the other way for NAND.
    if (BinOpcode == AArch64::BICwww_lsl || BinOpcode == AArch64::BICxxx_lsl)
      BuildMI(BB, dl, TII->get(BinOpcode), scratch)
        .addReg(incr).addReg(dest).addImm(0);
    else
      BuildMI(BB, dl, TII->get(BinOpcode), scratch)
        .addReg(dest).addReg(incr).addImm(0);
  }

  // From the stxr, the register is GPR32; from the cmp it's GPR32wsp
  unsigned stxr_status = MRI.createVirtualRegister(&AArch64::GPR32RegClass);
  MRI.constrainRegClass(stxr_status, &AArch64::GPR32wspRegClass);

  BuildMI(BB, dl, TII->get(strOpc), stxr_status).addReg(scratch).addReg(ptr);
  BuildMI(BB, dl, TII->get(AArch64::CBNZw))
    .addReg(stxr_status).addMBB(loopMBB);

  BB->addSuccessor(loopMBB);
  BB->addSuccessor(exitMBB);

  //  exitMBB:
  //   ...
  BB = exitMBB;

  MI->eraseFromParent();   // The instruction is gone now.

  return BB;
}

MachineBasicBlock *
AArch64TargetLowering::emitAtomicBinaryMinMax(MachineInstr *MI,
                                              MachineBasicBlock *BB,
                                              unsigned Size,
                                              unsigned CmpOp,
                                              A64CC::CondCodes Cond) const {
  const TargetInstrInfo *TII = getTargetMachine().getInstrInfo();

  const BasicBlock *LLVM_BB = BB->getBasicBlock();
  MachineFunction *MF = BB->getParent();
  MachineFunction::iterator It = BB;
  ++It;

  unsigned dest = MI->getOperand(0).getReg();
  unsigned ptr = MI->getOperand(1).getReg();
  unsigned incr = MI->getOperand(2).getReg();
  AtomicOrdering Ord = static_cast<AtomicOrdering>(MI->getOperand(3).getImm());

  unsigned oldval = dest;
  DebugLoc dl = MI->getDebugLoc();

  MachineRegisterInfo &MRI = BB->getParent()->getRegInfo();
  const TargetRegisterClass *TRC, *TRCsp;
  if (Size == 8) {
    TRC = &AArch64::GPR64RegClass;
    TRCsp = &AArch64::GPR64xspRegClass;
  } else {
    TRC = &AArch64::GPR32RegClass;
    TRCsp = &AArch64::GPR32wspRegClass;
  }

  unsigned ldrOpc, strOpc;
  getExclusiveOperation(Size, Ord, ldrOpc, strOpc);

  MachineBasicBlock *loopMBB = MF->CreateMachineBasicBlock(LLVM_BB);
  MachineBasicBlock *exitMBB = MF->CreateMachineBasicBlock(LLVM_BB);
  MF->insert(It, loopMBB);
  MF->insert(It, exitMBB);

  // Transfer the remainder of BB and its successor edges to exitMBB.
  exitMBB->splice(exitMBB->begin(), BB,
                  llvm::next(MachineBasicBlock::iterator(MI)),
                  BB->end());
  exitMBB->transferSuccessorsAndUpdatePHIs(BB);

  unsigned scratch = MRI.createVirtualRegister(TRC);
  MRI.constrainRegClass(scratch, TRCsp);

  //  thisMBB:
  //   ...
  //   fallthrough --> loopMBB
  BB->addSuccessor(loopMBB);

  //  loopMBB:
  //   ldxr dest, ptr
  //   cmp incr, dest (, sign extend if necessary)
  //   csel scratch, dest, incr, cond
  //   stxr stxr_status, scratch, ptr
  //   cbnz stxr_status, loopMBB
  //   fallthrough --> exitMBB
  BB = loopMBB;
  BuildMI(BB, dl, TII->get(ldrOpc), dest).addReg(ptr);

  // Build compare and cmov instructions.
  MRI.constrainRegClass(incr, TRCsp);
  BuildMI(BB, dl, TII->get(CmpOp))
    .addReg(incr).addReg(oldval).addImm(0);

  BuildMI(BB, dl, TII->get(Size == 8 ? AArch64::CSELxxxc : AArch64::CSELwwwc),
          scratch)
    .addReg(oldval).addReg(incr).addImm(Cond);

  unsigned stxr_status = MRI.createVirtualRegister(&AArch64::GPR32RegClass);
  MRI.constrainRegClass(stxr_status, &AArch64::GPR32wspRegClass);

  BuildMI(BB, dl, TII->get(strOpc), stxr_status)
    .addReg(scratch).addReg(ptr);
  BuildMI(BB, dl, TII->get(AArch64::CBNZw))
    .addReg(stxr_status).addMBB(loopMBB);

  BB->addSuccessor(loopMBB);
  BB->addSuccessor(exitMBB);

  //  exitMBB:
  //   ...
  BB = exitMBB;

  MI->eraseFromParent();   // The instruction is gone now.

  return BB;
}

MachineBasicBlock *
AArch64TargetLowering::emitAtomicCmpSwap(MachineInstr *MI,
                                         MachineBasicBlock *BB,
                                         unsigned Size) const {
  unsigned dest    = MI->getOperand(0).getReg();
  unsigned ptr     = MI->getOperand(1).getReg();
  unsigned oldval  = MI->getOperand(2).getReg();
  unsigned newval  = MI->getOperand(3).getReg();
  AtomicOrdering Ord = static_cast<AtomicOrdering>(MI->getOperand(4).getImm());
  const TargetInstrInfo *TII = getTargetMachine().getInstrInfo();
  DebugLoc dl = MI->getDebugLoc();

  MachineRegisterInfo &MRI = BB->getParent()->getRegInfo();
  const TargetRegisterClass *TRCsp;
  TRCsp = Size == 8 ? &AArch64::GPR64xspRegClass : &AArch64::GPR32wspRegClass;

  unsigned ldrOpc, strOpc;
  getExclusiveOperation(Size, Ord, ldrOpc, strOpc);

  MachineFunction *MF = BB->getParent();
  const BasicBlock *LLVM_BB = BB->getBasicBlock();
  MachineFunction::iterator It = BB;
  ++It; // insert the new blocks after the current block

  MachineBasicBlock *loop1MBB = MF->CreateMachineBasicBlock(LLVM_BB);
  MachineBasicBlock *loop2MBB = MF->CreateMachineBasicBlock(LLVM_BB);
  MachineBasicBlock *exitMBB = MF->CreateMachineBasicBlock(LLVM_BB);
  MF->insert(It, loop1MBB);
  MF->insert(It, loop2MBB);
  MF->insert(It, exitMBB);

  // Transfer the remainder of BB and its successor edges to exitMBB.
  exitMBB->splice(exitMBB->begin(), BB,
                  llvm::next(MachineBasicBlock::iterator(MI)),
                  BB->end());
  exitMBB->transferSuccessorsAndUpdatePHIs(BB);

  //  thisMBB:
  //   ...
  //   fallthrough --> loop1MBB
  BB->addSuccessor(loop1MBB);

  // loop1MBB:
  //   ldxr dest, [ptr]
  //   cmp dest, oldval
  //   b.ne exitMBB
  BB = loop1MBB;
  BuildMI(BB, dl, TII->get(ldrOpc), dest).addReg(ptr);

  unsigned CmpOp = Size == 8 ? AArch64::CMPxx_lsl : AArch64::CMPww_lsl;
  MRI.constrainRegClass(dest, TRCsp);
  BuildMI(BB, dl, TII->get(CmpOp))
    .addReg(dest).addReg(oldval).addImm(0);
  BuildMI(BB, dl, TII->get(AArch64::Bcc))
    .addImm(A64CC::NE).addMBB(exitMBB);
  BB->addSuccessor(loop2MBB);
  BB->addSuccessor(exitMBB);

  // loop2MBB:
  //   strex stxr_status, newval, [ptr]
  //   cbnz stxr_status, loop1MBB
  BB = loop2MBB;
  unsigned stxr_status = MRI.createVirtualRegister(&AArch64::GPR32RegClass);
  MRI.constrainRegClass(stxr_status, &AArch64::GPR32wspRegClass);

  BuildMI(BB, dl, TII->get(strOpc), stxr_status).addReg(newval).addReg(ptr);
  BuildMI(BB, dl, TII->get(AArch64::CBNZw))
    .addReg(stxr_status).addMBB(loop1MBB);
  BB->addSuccessor(loop1MBB);
  BB->addSuccessor(exitMBB);

  //  exitMBB:
  //   ...
  BB = exitMBB;

  MI->eraseFromParent();   // The instruction is gone now.

  return BB;
}

MachineBasicBlock *
AArch64TargetLowering::EmitF128CSEL(MachineInstr *MI,
                                    MachineBasicBlock *MBB) const {
  // We materialise the F128CSEL pseudo-instruction using conditional branches
  // and loads, giving an instruciton sequence like:
  //     str q0, [sp]
  //     b.ne IfTrue
  //     b Finish
  // IfTrue:
  //     str q1, [sp]
  // Finish:
  //     ldr q0, [sp]
  //
  // Using virtual registers would probably not be beneficial since COPY
  // instructions are expensive for f128 (there's no actual instruction to
  // implement them).
  //
  // An alternative would be to do an integer-CSEL on some address. E.g.:
  //     mov x0, sp
  //     add x1, sp, #16
  //     str q0, [x0]
  //     str q1, [x1]
  //     csel x0, x0, x1, ne
  //     ldr q0, [x0]
  //
  // It's unclear which approach is actually optimal.
  const TargetInstrInfo *TII = getTargetMachine().getInstrInfo();
  MachineFunction *MF = MBB->getParent();
  const BasicBlock *LLVM_BB = MBB->getBasicBlock();
  DebugLoc DL = MI->getDebugLoc();
  MachineFunction::iterator It = MBB;
  ++It;

  unsigned DestReg = MI->getOperand(0).getReg();
  unsigned IfTrueReg = MI->getOperand(1).getReg();
  unsigned IfFalseReg = MI->getOperand(2).getReg();
  unsigned CondCode = MI->getOperand(3).getImm();
  bool NZCVKilled = MI->getOperand(4).isKill();

  MachineBasicBlock *TrueBB = MF->CreateMachineBasicBlock(LLVM_BB);
  MachineBasicBlock *EndBB = MF->CreateMachineBasicBlock(LLVM_BB);
  MF->insert(It, TrueBB);
  MF->insert(It, EndBB);

  // Transfer rest of current basic-block to EndBB
  EndBB->splice(EndBB->begin(), MBB,
                llvm::next(MachineBasicBlock::iterator(MI)),
                MBB->end());
  EndBB->transferSuccessorsAndUpdatePHIs(MBB);

  // We need somewhere to store the f128 value needed.
  int ScratchFI = MF->getFrameInfo()->CreateSpillStackObject(16, 16);

  //     [... start of incoming MBB ...]
  //     str qIFFALSE, [sp]
  //     b.cc IfTrue
  //     b Done
  BuildMI(MBB, DL, TII->get(AArch64::LSFP128_STR))
    .addReg(IfFalseReg)
    .addFrameIndex(ScratchFI)
    .addImm(0);
  BuildMI(MBB, DL, TII->get(AArch64::Bcc))
    .addImm(CondCode)
    .addMBB(TrueBB);
  BuildMI(MBB, DL, TII->get(AArch64::Bimm))
    .addMBB(EndBB);
  MBB->addSuccessor(TrueBB);
  MBB->addSuccessor(EndBB);

  // IfTrue:
  //     str qIFTRUE, [sp]
  BuildMI(TrueBB, DL, TII->get(AArch64::LSFP128_STR))
    .addReg(IfTrueReg)
    .addFrameIndex(ScratchFI)
    .addImm(0);

  // Note: fallthrough. We can rely on LLVM adding a branch if it reorders the
  // blocks.
  TrueBB->addSuccessor(EndBB);

  // Done:
  //     ldr qDEST, [sp]
  //     [... rest of incoming MBB ...]
  if (!NZCVKilled)
    EndBB->addLiveIn(AArch64::NZCV);
  MachineInstr *StartOfEnd = EndBB->begin();
  BuildMI(*EndBB, StartOfEnd, DL, TII->get(AArch64::LSFP128_LDR), DestReg)
    .addFrameIndex(ScratchFI)
    .addImm(0);

  MI->eraseFromParent();
  return EndBB;
}

MachineBasicBlock *
AArch64TargetLowering::EmitInstrWithCustomInserter(MachineInstr *MI,
                                                 MachineBasicBlock *MBB) const {
  switch (MI->getOpcode()) {
  default: llvm_unreachable("Unhandled instruction with custom inserter");
  case AArch64::F128CSEL:
    return EmitF128CSEL(MI, MBB);
  case AArch64::ATOMIC_LOAD_ADD_I8:
    return emitAtomicBinary(MI, MBB, 1, AArch64::ADDwww_lsl);
  case AArch64::ATOMIC_LOAD_ADD_I16:
    return emitAtomicBinary(MI, MBB, 2, AArch64::ADDwww_lsl);
  case AArch64::ATOMIC_LOAD_ADD_I32:
    return emitAtomicBinary(MI, MBB, 4, AArch64::ADDwww_lsl);
  case AArch64::ATOMIC_LOAD_ADD_I64:
    return emitAtomicBinary(MI, MBB, 8, AArch64::ADDxxx_lsl);

  case AArch64::ATOMIC_LOAD_SUB_I8:
    return emitAtomicBinary(MI, MBB, 1, AArch64::SUBwww_lsl);
  case AArch64::ATOMIC_LOAD_SUB_I16:
    return emitAtomicBinary(MI, MBB, 2, AArch64::SUBwww_lsl);
  case AArch64::ATOMIC_LOAD_SUB_I32:
    return emitAtomicBinary(MI, MBB, 4, AArch64::SUBwww_lsl);
  case AArch64::ATOMIC_LOAD_SUB_I64:
    return emitAtomicBinary(MI, MBB, 8, AArch64::SUBxxx_lsl);

  case AArch64::ATOMIC_LOAD_AND_I8:
    return emitAtomicBinary(MI, MBB, 1, AArch64::ANDwww_lsl);
  case AArch64::ATOMIC_LOAD_AND_I16:
    return emitAtomicBinary(MI, MBB, 2, AArch64::ANDwww_lsl);
  case AArch64::ATOMIC_LOAD_AND_I32:
    return emitAtomicBinary(MI, MBB, 4, AArch64::ANDwww_lsl);
  case AArch64::ATOMIC_LOAD_AND_I64:
    return emitAtomicBinary(MI, MBB, 8, AArch64::ANDxxx_lsl);

  case AArch64::ATOMIC_LOAD_OR_I8:
    return emitAtomicBinary(MI, MBB, 1, AArch64::ORRwww_lsl);
  case AArch64::ATOMIC_LOAD_OR_I16:
    return emitAtomicBinary(MI, MBB, 2, AArch64::ORRwww_lsl);
  case AArch64::ATOMIC_LOAD_OR_I32:
    return emitAtomicBinary(MI, MBB, 4, AArch64::ORRwww_lsl);
  case AArch64::ATOMIC_LOAD_OR_I64:
    return emitAtomicBinary(MI, MBB, 8, AArch64::ORRxxx_lsl);

  case AArch64::ATOMIC_LOAD_XOR_I8:
    return emitAtomicBinary(MI, MBB, 1, AArch64::EORwww_lsl);
  case AArch64::ATOMIC_LOAD_XOR_I16:
    return emitAtomicBinary(MI, MBB, 2, AArch64::EORwww_lsl);
  case AArch64::ATOMIC_LOAD_XOR_I32:
    return emitAtomicBinary(MI, MBB, 4, AArch64::EORwww_lsl);
  case AArch64::ATOMIC_LOAD_XOR_I64:
    return emitAtomicBinary(MI, MBB, 8, AArch64::EORxxx_lsl);

  case AArch64::ATOMIC_LOAD_NAND_I8:
    return emitAtomicBinary(MI, MBB, 1, AArch64::BICwww_lsl);
  case AArch64::ATOMIC_LOAD_NAND_I16:
    return emitAtomicBinary(MI, MBB, 2, AArch64::BICwww_lsl);
  case AArch64::ATOMIC_LOAD_NAND_I32:
    return emitAtomicBinary(MI, MBB, 4, AArch64::BICwww_lsl);
  case AArch64::ATOMIC_LOAD_NAND_I64:
    return emitAtomicBinary(MI, MBB, 8, AArch64::BICxxx_lsl);

  case AArch64::ATOMIC_LOAD_MIN_I8:
    return emitAtomicBinaryMinMax(MI, MBB, 1, AArch64::CMPww_sxtb, A64CC::GT);
  case AArch64::ATOMIC_LOAD_MIN_I16:
    return emitAtomicBinaryMinMax(MI, MBB, 2, AArch64::CMPww_sxth, A64CC::GT);
  case AArch64::ATOMIC_LOAD_MIN_I32:
    return emitAtomicBinaryMinMax(MI, MBB, 4, AArch64::CMPww_lsl, A64CC::GT);
  case AArch64::ATOMIC_LOAD_MIN_I64:
    return emitAtomicBinaryMinMax(MI, MBB, 8, AArch64::CMPxx_lsl, A64CC::GT);

  case AArch64::ATOMIC_LOAD_MAX_I8:
    return emitAtomicBinaryMinMax(MI, MBB, 1, AArch64::CMPww_sxtb, A64CC::LT);
  case AArch64::ATOMIC_LOAD_MAX_I16:
    return emitAtomicBinaryMinMax(MI, MBB, 2, AArch64::CMPww_sxth, A64CC::LT);
  case AArch64::ATOMIC_LOAD_MAX_I32:
    return emitAtomicBinaryMinMax(MI, MBB, 4, AArch64::CMPww_lsl, A64CC::LT);
  case AArch64::ATOMIC_LOAD_MAX_I64:
    return emitAtomicBinaryMinMax(MI, MBB, 8, AArch64::CMPxx_lsl, A64CC::LT);

  case AArch64::ATOMIC_LOAD_UMIN_I8:
    return emitAtomicBinaryMinMax(MI, MBB, 1, AArch64::CMPww_uxtb, A64CC::HI);
  case AArch64::ATOMIC_LOAD_UMIN_I16:
    return emitAtomicBinaryMinMax(MI, MBB, 2, AArch64::CMPww_uxth, A64CC::HI);
  case AArch64::ATOMIC_LOAD_UMIN_I32:
    return emitAtomicBinaryMinMax(MI, MBB, 4, AArch64::CMPww_lsl, A64CC::HI);
  case AArch64::ATOMIC_LOAD_UMIN_I64:
    return emitAtomicBinaryMinMax(MI, MBB, 8, AArch64::CMPxx_lsl, A64CC::HI);

  case AArch64::ATOMIC_LOAD_UMAX_I8:
    return emitAtomicBinaryMinMax(MI, MBB, 1, AArch64::CMPww_uxtb, A64CC::LO);
  case AArch64::ATOMIC_LOAD_UMAX_I16:
    return emitAtomicBinaryMinMax(MI, MBB, 2, AArch64::CMPww_uxth, A64CC::LO);
  case AArch64::ATOMIC_LOAD_UMAX_I32:
    return emitAtomicBinaryMinMax(MI, MBB, 4, AArch64::CMPww_lsl, A64CC::LO);
  case AArch64::ATOMIC_LOAD_UMAX_I64:
    return emitAtomicBinaryMinMax(MI, MBB, 8, AArch64::CMPxx_lsl, A64CC::LO);

  case AArch64::ATOMIC_SWAP_I8:
    return emitAtomicBinary(MI, MBB, 1, 0);
  case AArch64::ATOMIC_SWAP_I16:
    return emitAtomicBinary(MI, MBB, 2, 0);
  case AArch64::ATOMIC_SWAP_I32:
    return emitAtomicBinary(MI, MBB, 4, 0);
  case AArch64::ATOMIC_SWAP_I64:
    return emitAtomicBinary(MI, MBB, 8, 0);

  case AArch64::ATOMIC_CMP_SWAP_I8:
    return emitAtomicCmpSwap(MI, MBB, 1);
  case AArch64::ATOMIC_CMP_SWAP_I16:
    return emitAtomicCmpSwap(MI, MBB, 2);
  case AArch64::ATOMIC_CMP_SWAP_I32:
    return emitAtomicCmpSwap(MI, MBB, 4);
  case AArch64::ATOMIC_CMP_SWAP_I64:
    return emitAtomicCmpSwap(MI, MBB, 8);
  }
}


const char *AArch64TargetLowering::getTargetNodeName(unsigned Opcode) const {
  switch (Opcode) {
  case AArch64ISD::BR_CC:          return "AArch64ISD::BR_CC";
  case AArch64ISD::Call:           return "AArch64ISD::Call";
  case AArch64ISD::FPMOV:          return "AArch64ISD::FPMOV";
  case AArch64ISD::GOTLoad:        return "AArch64ISD::GOTLoad";
  case AArch64ISD::BFI:            return "AArch64ISD::BFI";
  case AArch64ISD::EXTR:           return "AArch64ISD::EXTR";
  case AArch64ISD::Ret:            return "AArch64ISD::Ret";
  case AArch64ISD::SBFX:           return "AArch64ISD::SBFX";
  case AArch64ISD::SELECT_CC:      return "AArch64ISD::SELECT_CC";
  case AArch64ISD::SETCC:          return "AArch64ISD::SETCC";
  case AArch64ISD::TC_RETURN:      return "AArch64ISD::TC_RETURN";
  case AArch64ISD::THREAD_POINTER: return "AArch64ISD::THREAD_POINTER";
  case AArch64ISD::TLSDESCCALL:    return "AArch64ISD::TLSDESCCALL";
  case AArch64ISD::WrapperLarge:   return "AArch64ISD::WrapperLarge";
  case AArch64ISD::WrapperSmall:   return "AArch64ISD::WrapperSmall";

  default:                       return NULL;
  }
}

static const uint16_t AArch64FPRArgRegs[] = {
  AArch64::Q0, AArch64::Q1, AArch64::Q2, AArch64::Q3,
  AArch64::Q4, AArch64::Q5, AArch64::Q6, AArch64::Q7
};
static const unsigned NumFPRArgRegs = llvm::array_lengthof(AArch64FPRArgRegs);

static const uint16_t AArch64ArgRegs[] = {
  AArch64::X0, AArch64::X1, AArch64::X2, AArch64::X3,
  AArch64::X4, AArch64::X5, AArch64::X6, AArch64::X7
};
static const unsigned NumArgRegs = llvm::array_lengthof(AArch64ArgRegs);

static bool CC_AArch64NoMoreRegs(unsigned ValNo, MVT ValVT, MVT LocVT,
                                 CCValAssign::LocInfo LocInfo,
                                 ISD::ArgFlagsTy ArgFlags, CCState &State) {
  // Mark all remaining general purpose registers as allocated. We don't
  // backtrack: if (for example) an i128 gets put on the stack, no subsequent
  // i64 will go in registers (C.11).
  for (unsigned i = 0; i < NumArgRegs; ++i)
    State.AllocateReg(AArch64ArgRegs[i]);

  return false;
}

#include "AArch64GenCallingConv.inc"

CCAssignFn *AArch64TargetLowering::CCAssignFnForNode(CallingConv::ID CC) const {

  switch(CC) {
  default: llvm_unreachable("Unsupported calling convention");
  case CallingConv::Fast:
  case CallingConv::C:
    return CC_A64_APCS;
  }
}

void
AArch64TargetLowering::SaveVarArgRegisters(CCState &CCInfo, SelectionDAG &DAG,
                                           SDLoc DL, SDValue &Chain) const {
  MachineFunction &MF = DAG.getMachineFunction();
  MachineFrameInfo *MFI = MF.getFrameInfo();
  AArch64MachineFunctionInfo *FuncInfo
    = MF.getInfo<AArch64MachineFunctionInfo>();

  SmallVector<SDValue, 8> MemOps;

  unsigned FirstVariadicGPR = CCInfo.getFirstUnallocated(AArch64ArgRegs,
                                                         NumArgRegs);
  unsigned FirstVariadicFPR = CCInfo.getFirstUnallocated(AArch64FPRArgRegs,
                                                         NumFPRArgRegs);

  unsigned GPRSaveSize = 8 * (NumArgRegs - FirstVariadicGPR);
  int GPRIdx = 0;
  if (GPRSaveSize != 0) {
    GPRIdx = MFI->CreateStackObject(GPRSaveSize, 8, false);

    SDValue FIN = DAG.getFrameIndex(GPRIdx, getPointerTy());

    for (unsigned i = FirstVariadicGPR; i < NumArgRegs; ++i) {
      unsigned VReg = MF.addLiveIn(AArch64ArgRegs[i], &AArch64::GPR64RegClass);
      SDValue Val = DAG.getCopyFromReg(Chain, DL, VReg, MVT::i64);
      SDValue Store = DAG.getStore(Val.getValue(1), DL, Val, FIN,
                                   MachinePointerInfo::getStack(i * 8),
                                   false, false, 0);
      MemOps.push_back(Store);
      FIN = DAG.getNode(ISD::ADD, DL, getPointerTy(), FIN,
                        DAG.getConstant(8, getPointerTy()));
    }
  }

  unsigned FPRSaveSize = 16 * (NumFPRArgRegs - FirstVariadicFPR);
  int FPRIdx = 0;
  if (FPRSaveSize != 0) {
    FPRIdx = MFI->CreateStackObject(FPRSaveSize, 16, false);

    SDValue FIN = DAG.getFrameIndex(FPRIdx, getPointerTy());

    for (unsigned i = FirstVariadicFPR; i < NumFPRArgRegs; ++i) {
      unsigned VReg = MF.addLiveIn(AArch64FPRArgRegs[i],
                                   &AArch64::FPR128RegClass);
      SDValue Val = DAG.getCopyFromReg(Chain, DL, VReg, MVT::f128);
      SDValue Store = DAG.getStore(Val.getValue(1), DL, Val, FIN,
                                   MachinePointerInfo::getStack(i * 16),
                                   false, false, 0);
      MemOps.push_back(Store);
      FIN = DAG.getNode(ISD::ADD, DL, getPointerTy(), FIN,
                        DAG.getConstant(16, getPointerTy()));
    }
  }

  int StackIdx = MFI->CreateFixedObject(8, CCInfo.getNextStackOffset(), true);

  FuncInfo->setVariadicStackIdx(StackIdx);
  FuncInfo->setVariadicGPRIdx(GPRIdx);
  FuncInfo->setVariadicGPRSize(GPRSaveSize);
  FuncInfo->setVariadicFPRIdx(FPRIdx);
  FuncInfo->setVariadicFPRSize(FPRSaveSize);

  if (!MemOps.empty()) {
    Chain = DAG.getNode(ISD::TokenFactor, DL, MVT::Other, &MemOps[0],
                        MemOps.size());
  }
}


SDValue
AArch64TargetLowering::LowerFormalArguments(SDValue Chain,
                                      CallingConv::ID CallConv, bool isVarArg,
                                      const SmallVectorImpl<ISD::InputArg> &Ins,
                                      SDLoc dl, SelectionDAG &DAG,
                                      SmallVectorImpl<SDValue> &InVals) const {
  MachineFunction &MF = DAG.getMachineFunction();
  AArch64MachineFunctionInfo *FuncInfo
    = MF.getInfo<AArch64MachineFunctionInfo>();
  MachineFrameInfo *MFI = MF.getFrameInfo();
  bool TailCallOpt = MF.getTarget().Options.GuaranteedTailCallOpt;

  SmallVector<CCValAssign, 16> ArgLocs;
  CCState CCInfo(CallConv, isVarArg, DAG.getMachineFunction(),
                 getTargetMachine(), ArgLocs, *DAG.getContext());
  CCInfo.AnalyzeFormalArguments(Ins, CCAssignFnForNode(CallConv));

  SmallVector<SDValue, 16> ArgValues;

  SDValue ArgValue;
  for (unsigned i = 0, e = ArgLocs.size(); i != e; ++i) {
    CCValAssign &VA = ArgLocs[i];
    ISD::ArgFlagsTy Flags = Ins[i].Flags;

    if (Flags.isByVal()) {
      // Byval is used for small structs and HFAs in the PCS, but the system
      // should work in a non-compliant manner for larger structs.
      EVT PtrTy = getPointerTy();
      int Size = Flags.getByValSize();
      unsigned NumRegs = (Size + 7) / 8;

      unsigned FrameIdx = MFI->CreateFixedObject(8 * NumRegs,
                                                 VA.getLocMemOffset(),
                                                 false);
      SDValue FrameIdxN = DAG.getFrameIndex(FrameIdx, PtrTy);
      InVals.push_back(FrameIdxN);

      continue;
    } else if (VA.isRegLoc()) {
      MVT RegVT = VA.getLocVT();
      const TargetRegisterClass *RC = getRegClassFor(RegVT);
      unsigned Reg = MF.addLiveIn(VA.getLocReg(), RC);

      ArgValue = DAG.getCopyFromReg(Chain, dl, Reg, RegVT);
    } else { // VA.isRegLoc()
      assert(VA.isMemLoc());

      int FI = MFI->CreateFixedObject(VA.getLocVT().getSizeInBits()/8,
                                      VA.getLocMemOffset(), true);

      SDValue FIN = DAG.getFrameIndex(FI, getPointerTy());
      ArgValue = DAG.getLoad(VA.getLocVT(), dl, Chain, FIN,
                             MachinePointerInfo::getFixedStack(FI),
                             false, false, false, 0);


    }

    switch (VA.getLocInfo()) {
    default: llvm_unreachable("Unknown loc info!");
    case CCValAssign::Full: break;
    case CCValAssign::BCvt:
      ArgValue = DAG.getNode(ISD::BITCAST,dl, VA.getValVT(), ArgValue);
      break;
    case CCValAssign::SExt:
    case CCValAssign::ZExt:
    case CCValAssign::AExt: {
      unsigned DestSize = VA.getValVT().getSizeInBits();
      unsigned DestSubReg;

      switch (DestSize) {
      case 8: DestSubReg = AArch64::sub_8; break;
      case 16: DestSubReg = AArch64::sub_16; break;
      case 32: DestSubReg = AArch64::sub_32; break;
      case 64: DestSubReg = AArch64::sub_64; break;
      default: llvm_unreachable("Unexpected argument promotion");
      }

      ArgValue = SDValue(DAG.getMachineNode(TargetOpcode::EXTRACT_SUBREG, dl,
                                   VA.getValVT(), ArgValue,
                                   DAG.getTargetConstant(DestSubReg, MVT::i32)),
                         0);
      break;
    }
    }

    InVals.push_back(ArgValue);
  }

  if (isVarArg)
    SaveVarArgRegisters(CCInfo, DAG, dl, Chain);

  unsigned StackArgSize = CCInfo.getNextStackOffset();
  if (DoesCalleeRestoreStack(CallConv, TailCallOpt)) {
    // This is a non-standard ABI so by fiat I say we're allowed to make full
    // use of the stack area to be popped, which must be aligned to 16 bytes in
    // any case:
    StackArgSize = RoundUpToAlignment(StackArgSize, 16);

    // If we're expected to restore the stack (e.g. fastcc) then we'll be adding
    // a multiple of 16.
    FuncInfo->setArgumentStackToRestore(StackArgSize);

    // This realignment carries over to the available bytes below. Our own
    // callers will guarantee the space is free by giving an aligned value to
    // CALLSEQ_START.
  }
  // Even if we're not expected to free up the space, it's useful to know how
  // much is there while considering tail calls (because we can reuse it).
  FuncInfo->setBytesInStackArgArea(StackArgSize);

  return Chain;
}

SDValue
AArch64TargetLowering::LowerReturn(SDValue Chain,
                                   CallingConv::ID CallConv, bool isVarArg,
                                   const SmallVectorImpl<ISD::OutputArg> &Outs,
                                   const SmallVectorImpl<SDValue> &OutVals,
                                   SDLoc dl, SelectionDAG &DAG) const {
  // CCValAssign - represent the assignment of the return value to a location.
  SmallVector<CCValAssign, 16> RVLocs;

  // CCState - Info about the registers and stack slots.
  CCState CCInfo(CallConv, isVarArg, DAG.getMachineFunction(),
                 getTargetMachine(), RVLocs, *DAG.getContext());

  // Analyze outgoing return values.
  CCInfo.AnalyzeReturn(Outs, CCAssignFnForNode(CallConv));

  SDValue Flag;
  SmallVector<SDValue, 4> RetOps(1, Chain);

  for (unsigned i = 0, e = RVLocs.size(); i != e; ++i) {
    // PCS: "If the type, T, of the result of a function is such that
    // void func(T arg) would require that arg be passed as a value in a
    // register (or set of registers) according to the rules in 5.4, then the
    // result is returned in the same registers as would be used for such an
    // argument.
    //
    // Otherwise, the caller shall reserve a block of memory of sufficient
    // size and alignment to hold the result. The address of the memory block
    // shall be passed as an additional argument to the function in x8."
    //
    // This is implemented in two places. The register-return values are dealt
    // with here, more complex returns are passed as an sret parameter, which
    // means we don't have to worry about it during actual return.
    CCValAssign &VA = RVLocs[i];
    assert(VA.isRegLoc() && "Only register-returns should be created by PCS");


    SDValue Arg = OutVals[i];

    // There's no convenient note in the ABI about this as there is for normal
    // arguments, but it says return values are passed in the same registers as
    // an argument would be. I believe that includes the comments about
    // unspecified higher bits, putting the burden of widening on the *caller*
    // for return values.
    switch (VA.getLocInfo()) {
    default: llvm_unreachable("Unknown loc info");
    case CCValAssign::Full: break;
    case CCValAssign::SExt:
    case CCValAssign::ZExt:
    case CCValAssign::AExt:
      // Floating-point values should only be extended when they're going into
      // memory, which can't happen here so an integer extend is acceptable.
      Arg = DAG.getNode(ISD::ANY_EXTEND, dl, VA.getLocVT(), Arg);
      break;
    case CCValAssign::BCvt:
      Arg = DAG.getNode(ISD::BITCAST, dl, VA.getLocVT(), Arg);
      break;
    }

    Chain = DAG.getCopyToReg(Chain, dl, VA.getLocReg(), Arg, Flag);
    Flag = Chain.getValue(1);
    RetOps.push_back(DAG.getRegister(VA.getLocReg(), VA.getLocVT()));
  }

  RetOps[0] = Chain;  // Update chain.

  // Add the flag if we have it.
  if (Flag.getNode())
    RetOps.push_back(Flag);

  return DAG.getNode(AArch64ISD::Ret, dl, MVT::Other,
                     &RetOps[0], RetOps.size());
}

SDValue
AArch64TargetLowering::LowerCall(CallLoweringInfo &CLI,
                                 SmallVectorImpl<SDValue> &InVals) const {
  SelectionDAG &DAG                     = CLI.DAG;
  SDLoc &dl                             = CLI.DL;
  SmallVectorImpl<ISD::OutputArg> &Outs = CLI.Outs;
  SmallVectorImpl<SDValue> &OutVals     = CLI.OutVals;
  SmallVectorImpl<ISD::InputArg> &Ins   = CLI.Ins;
  SDValue Chain                         = CLI.Chain;
  SDValue Callee                        = CLI.Callee;
  bool &IsTailCall                      = CLI.IsTailCall;
  CallingConv::ID CallConv              = CLI.CallConv;
  bool IsVarArg                         = CLI.IsVarArg;

  MachineFunction &MF = DAG.getMachineFunction();
  AArch64MachineFunctionInfo *FuncInfo
    = MF.getInfo<AArch64MachineFunctionInfo>();
  bool TailCallOpt = MF.getTarget().Options.GuaranteedTailCallOpt;
  bool IsStructRet = !Outs.empty() && Outs[0].Flags.isSRet();
  bool IsSibCall = false;

  if (IsTailCall) {
    IsTailCall = IsEligibleForTailCallOptimization(Callee, CallConv,
                    IsVarArg, IsStructRet, MF.getFunction()->hasStructRetAttr(),
                                                   Outs, OutVals, Ins, DAG);

    // A sibling call is one where we're under the usual C ABI and not planning
    // to change that but can still do a tail call:
    if (!TailCallOpt && IsTailCall)
      IsSibCall = true;
  }

  SmallVector<CCValAssign, 16> ArgLocs;
  CCState CCInfo(CallConv, IsVarArg, DAG.getMachineFunction(),
                 getTargetMachine(), ArgLocs, *DAG.getContext());
  CCInfo.AnalyzeCallOperands(Outs, CCAssignFnForNode(CallConv));

  // On AArch64 (and all other architectures I'm aware of) the most this has to
  // do is adjust the stack pointer.
  unsigned NumBytes = RoundUpToAlignment(CCInfo.getNextStackOffset(), 16);
  if (IsSibCall) {
    // Since we're not changing the ABI to make this a tail call, the memory
    // operands are already available in the caller's incoming argument space.
    NumBytes = 0;
  }

  // FPDiff is the byte offset of the call's argument area from the callee's.
  // Stores to callee stack arguments will be placed in FixedStackSlots offset
  // by this amount for a tail call. In a sibling call it must be 0 because the
  // caller will deallocate the entire stack and the callee still expects its
  // arguments to begin at SP+0. Completely unused for non-tail calls.
  int FPDiff = 0;

  if (IsTailCall && !IsSibCall) {
    unsigned NumReusableBytes = FuncInfo->getBytesInStackArgArea();

    // FPDiff will be negative if this tail call requires more space than we
    // would automatically have in our incoming argument space. Positive if we
    // can actually shrink the stack.
    FPDiff = NumReusableBytes - NumBytes;

    // The stack pointer must be 16-byte aligned at all times it's used for a
    // memory operation, which in practice means at *all* times and in
    // particular across call boundaries. Therefore our own arguments started at
    // a 16-byte aligned SP and the delta applied for the tail call should
    // satisfy the same constraint.
    assert(FPDiff % 16 == 0 && "unaligned stack on tail call");
  }

  if (!IsSibCall)
    Chain = DAG.getCALLSEQ_START(Chain, DAG.getIntPtrConstant(NumBytes, true),
                                 dl);

  SDValue StackPtr = DAG.getCopyFromReg(Chain, dl, AArch64::XSP,
                                        getPointerTy());

  SmallVector<SDValue, 8> MemOpChains;
  SmallVector<std::pair<unsigned, SDValue>, 8> RegsToPass;

  for (unsigned i = 0, e = ArgLocs.size(); i != e; ++i) {
    CCValAssign &VA = ArgLocs[i];
    ISD::ArgFlagsTy Flags = Outs[i].Flags;
    SDValue Arg = OutVals[i];

    // Callee does the actual widening, so all extensions just use an implicit
    // definition of the rest of the Loc. Aesthetically, this would be nicer as
    // an ANY_EXTEND, but that isn't valid for floating-point types and this
    // alternative works on integer types too.
    switch (VA.getLocInfo()) {
    default: llvm_unreachable("Unknown loc info!");
    case CCValAssign::Full: break;
    case CCValAssign::SExt:
    case CCValAssign::ZExt:
    case CCValAssign::AExt: {
      unsigned SrcSize = VA.getValVT().getSizeInBits();
      unsigned SrcSubReg;

      switch (SrcSize) {
      case 8: SrcSubReg = AArch64::sub_8; break;
      case 16: SrcSubReg = AArch64::sub_16; break;
      case 32: SrcSubReg = AArch64::sub_32; break;
      case 64: SrcSubReg = AArch64::sub_64; break;
      default: llvm_unreachable("Unexpected argument promotion");
      }

      Arg = SDValue(DAG.getMachineNode(TargetOpcode::INSERT_SUBREG, dl,
                                    VA.getLocVT(),
                                    DAG.getUNDEF(VA.getLocVT()),
                                    Arg,
                                    DAG.getTargetConstant(SrcSubReg, MVT::i32)),
                    0);

      break;
    }
    case CCValAssign::BCvt:
      Arg = DAG.getNode(ISD::BITCAST, dl, VA.getLocVT(), Arg);
      break;
    }

    if (VA.isRegLoc()) {
      // A normal register (sub-) argument. For now we just note it down because
      // we want to copy things into registers as late as possible to avoid
      // register-pressure (and possibly worse).
      RegsToPass.push_back(std::make_pair(VA.getLocReg(), Arg));
      continue;
    }

    assert(VA.isMemLoc() && "unexpected argument location");

    SDValue DstAddr;
    MachinePointerInfo DstInfo;
    if (IsTailCall) {
      uint32_t OpSize = Flags.isByVal() ? Flags.getByValSize() :
                                          VA.getLocVT().getSizeInBits();
      OpSize = (OpSize + 7) / 8;
      int32_t Offset = VA.getLocMemOffset() + FPDiff;
      int FI = MF.getFrameInfo()->CreateFixedObject(OpSize, Offset, true);

      DstAddr = DAG.getFrameIndex(FI, getPointerTy());
      DstInfo = MachinePointerInfo::getFixedStack(FI);

      // Make sure any stack arguments overlapping with where we're storing are
      // loaded before this eventual operation. Otherwise they'll be clobbered.
      Chain = addTokenForArgument(Chain, DAG, MF.getFrameInfo(), FI);
    } else {
      SDValue PtrOff = DAG.getIntPtrConstant(VA.getLocMemOffset());

      DstAddr = DAG.getNode(ISD::ADD, dl, getPointerTy(), StackPtr, PtrOff);
      DstInfo = MachinePointerInfo::getStack(VA.getLocMemOffset());
    }

    if (Flags.isByVal()) {
      SDValue SizeNode = DAG.getConstant(Flags.getByValSize(), MVT::i64);
      SDValue Cpy = DAG.getMemcpy(Chain, dl, DstAddr, Arg, SizeNode,
                                  Flags.getByValAlign(),
                                  /*isVolatile = */ false,
                                  /*alwaysInline = */ false,
                                  DstInfo, MachinePointerInfo(0));
      MemOpChains.push_back(Cpy);
    } else {
      // Normal stack argument, put it where it's needed.
      SDValue Store = DAG.getStore(Chain, dl, Arg, DstAddr, DstInfo,
                                   false, false, 0);
      MemOpChains.push_back(Store);
    }
  }

  // The loads and stores generated above shouldn't clash with each
  // other. Combining them with this TokenFactor notes that fact for the rest of
  // the backend.
  if (!MemOpChains.empty())
    Chain = DAG.getNode(ISD::TokenFactor, dl, MVT::Other,
                        &MemOpChains[0], MemOpChains.size());

  // Most of the rest of the instructions need to be glued together; we don't
  // want assignments to actual registers used by a call to be rearranged by a
  // well-meaning scheduler.
  SDValue InFlag;

  for (unsigned i = 0, e = RegsToPass.size(); i != e; ++i) {
    Chain = DAG.getCopyToReg(Chain, dl, RegsToPass[i].first,
                             RegsToPass[i].second, InFlag);
    InFlag = Chain.getValue(1);
  }

  // The linker is responsible for inserting veneers when necessary to put a
  // function call destination in range, so we don't need to bother with a
  // wrapper here.
  if (GlobalAddressSDNode *G = dyn_cast<GlobalAddressSDNode>(Callee)) {
    const GlobalValue *GV = G->getGlobal();
    Callee = DAG.getTargetGlobalAddress(GV, dl, getPointerTy());
  } else if (ExternalSymbolSDNode *S = dyn_cast<ExternalSymbolSDNode>(Callee)) {
    const char *Sym = S->getSymbol();
    Callee = DAG.getTargetExternalSymbol(Sym, getPointerTy());
  }

  // We don't usually want to end the call-sequence here because we would tidy
  // the frame up *after* the call, however in the ABI-changing tail-call case
  // we've carefully laid out the parameters so that when sp is reset they'll be
  // in the correct location.
  if (IsTailCall && !IsSibCall) {
    Chain = DAG.getCALLSEQ_END(Chain, DAG.getIntPtrConstant(NumBytes, true),
                               DAG.getIntPtrConstant(0, true), InFlag, dl);
    InFlag = Chain.getValue(1);
  }

  // We produce the following DAG scheme for the actual call instruction:
  //     (AArch64Call Chain, Callee, reg1, ..., regn, preserveMask, inflag?
  //
  // Most arguments aren't going to be used and just keep the values live as
  // far as LLVM is concerned. It's expected to be selected as simply "bl
  // callee" (for a direct, non-tail call).
  std::vector<SDValue> Ops;
  Ops.push_back(Chain);
  Ops.push_back(Callee);

  if (IsTailCall) {
    // Each tail call may have to adjust the stack by a different amount, so
    // this information must travel along with the operation for eventual
    // consumption by emitEpilogue.
    Ops.push_back(DAG.getTargetConstant(FPDiff, MVT::i32));
  }

  for (unsigned i = 0, e = RegsToPass.size(); i != e; ++i)
    Ops.push_back(DAG.getRegister(RegsToPass[i].first,
                                  RegsToPass[i].second.getValueType()));


  // Add a register mask operand representing the call-preserved registers. This
  // is used later in codegen to constrain register-allocation.
  const TargetRegisterInfo *TRI = getTargetMachine().getRegisterInfo();
  const uint32_t *Mask = TRI->getCallPreservedMask(CallConv);
  assert(Mask && "Missing call preserved mask for calling convention");
  Ops.push_back(DAG.getRegisterMask(Mask));

  // If we needed glue, put it in as the last argument.
  if (InFlag.getNode())
    Ops.push_back(InFlag);

  SDVTList NodeTys = DAG.getVTList(MVT::Other, MVT::Glue);

  if (IsTailCall) {
    return DAG.getNode(AArch64ISD::TC_RETURN, dl, NodeTys, &Ops[0], Ops.size());
  }

  Chain = DAG.getNode(AArch64ISD::Call, dl, NodeTys, &Ops[0], Ops.size());
  InFlag = Chain.getValue(1);

  // Now we can reclaim the stack, just as well do it before working out where
  // our return value is.
  if (!IsSibCall) {
    uint64_t CalleePopBytes
      = DoesCalleeRestoreStack(CallConv, TailCallOpt) ? NumBytes : 0;

    Chain = DAG.getCALLSEQ_END(Chain, DAG.getIntPtrConstant(NumBytes, true),
                               DAG.getIntPtrConstant(CalleePopBytes, true),
                               InFlag, dl);
    InFlag = Chain.getValue(1);
  }

  return LowerCallResult(Chain, InFlag, CallConv,
                         IsVarArg, Ins, dl, DAG, InVals);
}

SDValue
AArch64TargetLowering::LowerCallResult(SDValue Chain, SDValue InFlag,
                                      CallingConv::ID CallConv, bool IsVarArg,
                                      const SmallVectorImpl<ISD::InputArg> &Ins,
                                      SDLoc dl, SelectionDAG &DAG,
                                      SmallVectorImpl<SDValue> &InVals) const {
  // Assign locations to each value returned by this call.
  SmallVector<CCValAssign, 16> RVLocs;
  CCState CCInfo(CallConv, IsVarArg, DAG.getMachineFunction(),
                 getTargetMachine(), RVLocs, *DAG.getContext());
  CCInfo.AnalyzeCallResult(Ins, CCAssignFnForNode(CallConv));

  for (unsigned i = 0; i != RVLocs.size(); ++i) {
    CCValAssign VA = RVLocs[i];

    // Return values that are too big to fit into registers should use an sret
    // pointer, so this can be a lot simpler than the main argument code.
    assert(VA.isRegLoc() && "Memory locations not expected for call return");

    SDValue Val = DAG.getCopyFromReg(Chain, dl, VA.getLocReg(), VA.getLocVT(),
                                     InFlag);
    Chain = Val.getValue(1);
    InFlag = Val.getValue(2);

    switch (VA.getLocInfo()) {
    default: llvm_unreachable("Unknown loc info!");
    case CCValAssign::Full: break;
    case CCValAssign::BCvt:
      Val = DAG.getNode(ISD::BITCAST, dl, VA.getValVT(), Val);
      break;
    case CCValAssign::ZExt:
    case CCValAssign::SExt:
    case CCValAssign::AExt:
      // Floating-point arguments only get extended/truncated if they're going
      // in memory, so using the integer operation is acceptable here.
      Val = DAG.getNode(ISD::TRUNCATE, dl, VA.getValVT(), Val);
      break;
    }

    InVals.push_back(Val);
  }

  return Chain;
}

bool
AArch64TargetLowering::IsEligibleForTailCallOptimization(SDValue Callee,
                                    CallingConv::ID CalleeCC,
                                    bool IsVarArg,
                                    bool IsCalleeStructRet,
                                    bool IsCallerStructRet,
                                    const SmallVectorImpl<ISD::OutputArg> &Outs,
                                    const SmallVectorImpl<SDValue> &OutVals,
                                    const SmallVectorImpl<ISD::InputArg> &Ins,
                                    SelectionDAG& DAG) const {

  // For CallingConv::C this function knows whether the ABI needs
  // changing. That's not true for other conventions so they will have to opt in
  // manually.
  if (!IsTailCallConvention(CalleeCC) && CalleeCC != CallingConv::C)
    return false;

  const MachineFunction &MF = DAG.getMachineFunction();
  const Function *CallerF = MF.getFunction();
  CallingConv::ID CallerCC = CallerF->getCallingConv();
  bool CCMatch = CallerCC == CalleeCC;

  // Byval parameters hand the function a pointer directly into the stack area
  // we want to reuse during a tail call. Working around this *is* possible (see
  // X86) but less efficient and uglier in LowerCall.
  for (Function::const_arg_iterator i = CallerF->arg_begin(),
         e = CallerF->arg_end(); i != e; ++i)
    if (i->hasByValAttr())
      return false;

  if (getTargetMachine().Options.GuaranteedTailCallOpt) {
    if (IsTailCallConvention(CalleeCC) && CCMatch)
      return true;
    return false;
  }

  // Now we search for cases where we can use a tail call without changing the
  // ABI. Sibcall is used in some places (particularly gcc) to refer to this
  // concept.

  // I want anyone implementing a new calling convention to think long and hard
  // about this assert.
  assert((!IsVarArg || CalleeCC == CallingConv::C)
         && "Unexpected variadic calling convention");

  if (IsVarArg && !Outs.empty()) {
    // At least two cases here: if caller is fastcc then we can't have any
    // memory arguments (we'd be expected to clean up the stack afterwards). If
    // caller is C then we could potentially use its argument area.

    // FIXME: for now we take the most conservative of these in both cases:
    // disallow all variadic memory operands.
    SmallVector<CCValAssign, 16> ArgLocs;
    CCState CCInfo(CalleeCC, IsVarArg, DAG.getMachineFunction(),
                   getTargetMachine(), ArgLocs, *DAG.getContext());

    CCInfo.AnalyzeCallOperands(Outs, CCAssignFnForNode(CalleeCC));
    for (unsigned i = 0, e = ArgLocs.size(); i != e; ++i)
      if (!ArgLocs[i].isRegLoc())
        return false;
  }

  // If the calling conventions do not match, then we'd better make sure the
  // results are returned in the same way as what the caller expects.
  if (!CCMatch) {
    SmallVector<CCValAssign, 16> RVLocs1;
    CCState CCInfo1(CalleeCC, false, DAG.getMachineFunction(),
                    getTargetMachine(), RVLocs1, *DAG.getContext());
    CCInfo1.AnalyzeCallResult(Ins, CCAssignFnForNode(CalleeCC));

    SmallVector<CCValAssign, 16> RVLocs2;
    CCState CCInfo2(CallerCC, false, DAG.getMachineFunction(),
                    getTargetMachine(), RVLocs2, *DAG.getContext());
    CCInfo2.AnalyzeCallResult(Ins, CCAssignFnForNode(CallerCC));

    if (RVLocs1.size() != RVLocs2.size())
      return false;
    for (unsigned i = 0, e = RVLocs1.size(); i != e; ++i) {
      if (RVLocs1[i].isRegLoc() != RVLocs2[i].isRegLoc())
        return false;
      if (RVLocs1[i].getLocInfo() != RVLocs2[i].getLocInfo())
        return false;
      if (RVLocs1[i].isRegLoc()) {
        if (RVLocs1[i].getLocReg() != RVLocs2[i].getLocReg())
          return false;
      } else {
        if (RVLocs1[i].getLocMemOffset() != RVLocs2[i].getLocMemOffset())
          return false;
      }
    }
  }

  // Nothing more to check if the callee is taking no arguments
  if (Outs.empty())
    return true;

  SmallVector<CCValAssign, 16> ArgLocs;
  CCState CCInfo(CalleeCC, IsVarArg, DAG.getMachineFunction(),
                 getTargetMachine(), ArgLocs, *DAG.getContext());

  CCInfo.AnalyzeCallOperands(Outs, CCAssignFnForNode(CalleeCC));

  const AArch64MachineFunctionInfo *FuncInfo
    = MF.getInfo<AArch64MachineFunctionInfo>();

  // If the stack arguments for this call would fit into our own save area then
  // the call can be made tail.
  return CCInfo.getNextStackOffset() <= FuncInfo->getBytesInStackArgArea();
}

bool AArch64TargetLowering::DoesCalleeRestoreStack(CallingConv::ID CallCC,
                                                   bool TailCallOpt) const {
  return CallCC == CallingConv::Fast && TailCallOpt;
}

bool AArch64TargetLowering::IsTailCallConvention(CallingConv::ID CallCC) const {
  return CallCC == CallingConv::Fast;
}

SDValue AArch64TargetLowering::addTokenForArgument(SDValue Chain,
                                                   SelectionDAG &DAG,
                                                   MachineFrameInfo *MFI,
                                                   int ClobberedFI) const {
  SmallVector<SDValue, 8> ArgChains;
  int64_t FirstByte = MFI->getObjectOffset(ClobberedFI);
  int64_t LastByte = FirstByte + MFI->getObjectSize(ClobberedFI) - 1;

  // Include the original chain at the beginning of the list. When this is
  // used by target LowerCall hooks, this helps legalize find the
  // CALLSEQ_BEGIN node.
  ArgChains.push_back(Chain);

  // Add a chain value for each stack argument corresponding
  for (SDNode::use_iterator U = DAG.getEntryNode().getNode()->use_begin(),
         UE = DAG.getEntryNode().getNode()->use_end(); U != UE; ++U)
    if (LoadSDNode *L = dyn_cast<LoadSDNode>(*U))
      if (FrameIndexSDNode *FI = dyn_cast<FrameIndexSDNode>(L->getBasePtr()))
        if (FI->getIndex() < 0) {
          int64_t InFirstByte = MFI->getObjectOffset(FI->getIndex());
          int64_t InLastByte = InFirstByte;
          InLastByte += MFI->getObjectSize(FI->getIndex()) - 1;

          if ((InFirstByte <= FirstByte && FirstByte <= InLastByte) ||
              (FirstByte <= InFirstByte && InFirstByte <= LastByte))
            ArgChains.push_back(SDValue(L, 1));
        }

   // Build a tokenfactor for all the chains.
   return DAG.getNode(ISD::TokenFactor, SDLoc(Chain), MVT::Other,
                      &ArgChains[0], ArgChains.size());
}

static A64CC::CondCodes IntCCToA64CC(ISD::CondCode CC) {
  switch (CC) {
  case ISD::SETEQ:  return A64CC::EQ;
  case ISD::SETGT:  return A64CC::GT;
  case ISD::SETGE:  return A64CC::GE;
  case ISD::SETLT:  return A64CC::LT;
  case ISD::SETLE:  return A64CC::LE;
  case ISD::SETNE:  return A64CC::NE;
  case ISD::SETUGT: return A64CC::HI;
  case ISD::SETUGE: return A64CC::HS;
  case ISD::SETULT: return A64CC::LO;
  case ISD::SETULE: return A64CC::LS;
  default: llvm_unreachable("Unexpected condition code");
  }
}

bool AArch64TargetLowering::isLegalICmpImmediate(int64_t Val) const {
  // icmp is implemented using adds/subs immediate, which take an unsigned
  // 12-bit immediate, optionally shifted left by 12 bits.

  // Symmetric by using adds/subs
  if (Val < 0)
    Val = -Val;

  return (Val & ~0xfff) == 0 || (Val & ~0xfff000) == 0;
}

SDValue AArch64TargetLowering::getSelectableIntSetCC(SDValue LHS, SDValue RHS,
                                        ISD::CondCode CC, SDValue &A64cc,
                                        SelectionDAG &DAG, SDLoc &dl) const {
  if (ConstantSDNode *RHSC = dyn_cast<ConstantSDNode>(RHS.getNode())) {
    int64_t C = 0;
    EVT VT = RHSC->getValueType(0);
    bool knownInvalid = false;

    // I'm not convinced the rest of LLVM handles these edge cases properly, but
    // we can at least get it right.
    if (isSignedIntSetCC(CC)) {
      C = RHSC->getSExtValue();
    } else if (RHSC->getZExtValue() > INT64_MAX) {
      // A 64-bit constant not representable by a signed 64-bit integer is far
      // too big to fit into a SUBS immediate anyway.
      knownInvalid = true;
    } else {
      C = RHSC->getZExtValue();
    }

    if (!knownInvalid && !isLegalICmpImmediate(C)) {
      // Constant does not fit, try adjusting it by one?
      switch (CC) {
      default: break;
      case ISD::SETLT:
      case ISD::SETGE:
        if (isLegalICmpImmediate(C-1)) {
          CC = (CC == ISD::SETLT) ? ISD::SETLE : ISD::SETGT;
          RHS = DAG.getConstant(C-1, VT);
        }
        break;
      case ISD::SETULT:
      case ISD::SETUGE:
        if (isLegalICmpImmediate(C-1)) {
          CC = (CC == ISD::SETULT) ? ISD::SETULE : ISD::SETUGT;
          RHS = DAG.getConstant(C-1, VT);
        }
        break;
      case ISD::SETLE:
      case ISD::SETGT:
        if (isLegalICmpImmediate(C+1)) {
          CC = (CC == ISD::SETLE) ? ISD::SETLT : ISD::SETGE;
          RHS = DAG.getConstant(C+1, VT);
        }
        break;
      case ISD::SETULE:
      case ISD::SETUGT:
        if (isLegalICmpImmediate(C+1)) {
          CC = (CC == ISD::SETULE) ? ISD::SETULT : ISD::SETUGE;
          RHS = DAG.getConstant(C+1, VT);
        }
        break;
      }
    }
  }

  A64CC::CondCodes CondCode = IntCCToA64CC(CC);
  A64cc = DAG.getConstant(CondCode, MVT::i32);
  return DAG.getNode(AArch64ISD::SETCC, dl, MVT::i32, LHS, RHS,
                     DAG.getCondCode(CC));
}

static A64CC::CondCodes FPCCToA64CC(ISD::CondCode CC,
                                    A64CC::CondCodes &Alternative) {
  A64CC::CondCodes CondCode = A64CC::Invalid;
  Alternative = A64CC::Invalid;

  switch (CC) {
  default: llvm_unreachable("Unknown FP condition!");
  case ISD::SETEQ:
  case ISD::SETOEQ: CondCode = A64CC::EQ; break;
  case ISD::SETGT:
  case ISD::SETOGT: CondCode = A64CC::GT; break;
  case ISD::SETGE:
  case ISD::SETOGE: CondCode = A64CC::GE; break;
  case ISD::SETOLT: CondCode = A64CC::MI; break;
  case ISD::SETOLE: CondCode = A64CC::LS; break;
  case ISD::SETONE: CondCode = A64CC::MI; Alternative = A64CC::GT; break;
  case ISD::SETO:   CondCode = A64CC::VC; break;
  case ISD::SETUO:  CondCode = A64CC::VS; break;
  case ISD::SETUEQ: CondCode = A64CC::EQ; Alternative = A64CC::VS; break;
  case ISD::SETUGT: CondCode = A64CC::HI; break;
  case ISD::SETUGE: CondCode = A64CC::PL; break;
  case ISD::SETLT:
  case ISD::SETULT: CondCode = A64CC::LT; break;
  case ISD::SETLE:
  case ISD::SETULE: CondCode = A64CC::LE; break;
  case ISD::SETNE:
  case ISD::SETUNE: CondCode = A64CC::NE; break;
  }
  return CondCode;
}

SDValue
AArch64TargetLowering::LowerBlockAddress(SDValue Op, SelectionDAG &DAG) const {
  SDLoc DL(Op);
  EVT PtrVT = getPointerTy();
  const BlockAddress *BA = cast<BlockAddressSDNode>(Op)->getBlockAddress();

  switch(getTargetMachine().getCodeModel()) {
  case CodeModel::Small:
    // The most efficient code is PC-relative anyway for the small memory model,
    // so we don't need to worry about relocation model.
    return DAG.getNode(AArch64ISD::WrapperSmall, DL, PtrVT,
                       DAG.getTargetBlockAddress(BA, PtrVT, 0,
                                                 AArch64II::MO_NO_FLAG),
                       DAG.getTargetBlockAddress(BA, PtrVT, 0,
                                                 AArch64II::MO_LO12),
                       DAG.getConstant(/*Alignment=*/ 4, MVT::i32));
  case CodeModel::Large:
    return DAG.getNode(
      AArch64ISD::WrapperLarge, DL, PtrVT,
      DAG.getTargetBlockAddress(BA, PtrVT, 0, AArch64II::MO_ABS_G3),
      DAG.getTargetBlockAddress(BA, PtrVT, 0, AArch64II::MO_ABS_G2_NC),
      DAG.getTargetBlockAddress(BA, PtrVT, 0, AArch64II::MO_ABS_G1_NC),
      DAG.getTargetBlockAddress(BA, PtrVT, 0, AArch64II::MO_ABS_G0_NC));
  default:
    llvm_unreachable("Only small and large code models supported now");
  }
}


// (BRCOND chain, val, dest)
SDValue
AArch64TargetLowering::LowerBRCOND(SDValue Op, SelectionDAG &DAG) const {
  SDLoc dl(Op);
  SDValue Chain = Op.getOperand(0);
  SDValue TheBit = Op.getOperand(1);
  SDValue DestBB = Op.getOperand(2);

  // AArch64 BooleanContents is the default UndefinedBooleanContent, which means
  // that as the consumer we are responsible for ignoring rubbish in higher
  // bits.
  TheBit = DAG.getNode(ISD::AND, dl, MVT::i32, TheBit,
                       DAG.getConstant(1, MVT::i32));

  SDValue A64CMP = DAG.getNode(AArch64ISD::SETCC, dl, MVT::i32, TheBit,
                               DAG.getConstant(0, TheBit.getValueType()),
                               DAG.getCondCode(ISD::SETNE));

  return DAG.getNode(AArch64ISD::BR_CC, dl, MVT::Other, Chain,
                     A64CMP, DAG.getConstant(A64CC::NE, MVT::i32),
                     DestBB);
}

// (BR_CC chain, condcode, lhs, rhs, dest)
SDValue
AArch64TargetLowering::LowerBR_CC(SDValue Op, SelectionDAG &DAG) const {
  SDLoc dl(Op);
  SDValue Chain = Op.getOperand(0);
  ISD::CondCode CC = cast<CondCodeSDNode>(Op.getOperand(1))->get();
  SDValue LHS = Op.getOperand(2);
  SDValue RHS = Op.getOperand(3);
  SDValue DestBB = Op.getOperand(4);

  if (LHS.getValueType() == MVT::f128) {
    // f128 comparisons are lowered to runtime calls by a routine which sets
    // LHS, RHS and CC appropriately for the rest of this function to continue.
    softenSetCCOperands(DAG, MVT::f128, LHS, RHS, CC, dl);

    // If softenSetCCOperands returned a scalar, we need to compare the result
    // against zero to select between true and false values.
    if (RHS.getNode() == 0) {
      RHS = DAG.getConstant(0, LHS.getValueType());
      CC = ISD::SETNE;
    }
  }

  if (LHS.getValueType().isInteger()) {
    SDValue A64cc;

    // Integers are handled in a separate function because the combinations of
    // immediates and tests can get hairy and we may want to fiddle things.
    SDValue CmpOp = getSelectableIntSetCC(LHS, RHS, CC, A64cc, DAG, dl);

    return DAG.getNode(AArch64ISD::BR_CC, dl, MVT::Other,
                       Chain, CmpOp, A64cc, DestBB);
  }

  // Note that some LLVM floating-point CondCodes can't be lowered to a single
  // conditional branch, hence FPCCToA64CC can set a second test, where either
  // passing is sufficient.
  A64CC::CondCodes CondCode, Alternative = A64CC::Invalid;
  CondCode = FPCCToA64CC(CC, Alternative);
  SDValue A64cc = DAG.getConstant(CondCode, MVT::i32);
  SDValue SetCC = DAG.getNode(AArch64ISD::SETCC, dl, MVT::i32, LHS, RHS,
                              DAG.getCondCode(CC));
  SDValue A64BR_CC = DAG.getNode(AArch64ISD::BR_CC, dl, MVT::Other,
                                 Chain, SetCC, A64cc, DestBB);

  if (Alternative != A64CC::Invalid) {
    A64cc = DAG.getConstant(Alternative, MVT::i32);
    A64BR_CC = DAG.getNode(AArch64ISD::BR_CC, dl, MVT::Other,
                           A64BR_CC, SetCC, A64cc, DestBB);

  }

  return A64BR_CC;
}

SDValue
AArch64TargetLowering::LowerF128ToCall(SDValue Op, SelectionDAG &DAG,
                                       RTLIB::Libcall Call) const {
  ArgListTy Args;
  ArgListEntry Entry;
  for (unsigned i = 0, e = Op->getNumOperands(); i != e; ++i) {
    EVT ArgVT = Op.getOperand(i).getValueType();
    Type *ArgTy = ArgVT.getTypeForEVT(*DAG.getContext());
    Entry.Node = Op.getOperand(i); Entry.Ty = ArgTy;
    Entry.isSExt = false;
    Entry.isZExt = false;
    Args.push_back(Entry);
  }
  SDValue Callee = DAG.getExternalSymbol(getLibcallName(Call), getPointerTy());

  Type *RetTy = Op.getValueType().getTypeForEVT(*DAG.getContext());

  // By default, the input chain to this libcall is the entry node of the
  // function. If the libcall is going to be emitted as a tail call then
  // isUsedByReturnOnly will change it to the right chain if the return
  // node which is being folded has a non-entry input chain.
  SDValue InChain = DAG.getEntryNode();

  // isTailCall may be true since the callee does not reference caller stack
  // frame. Check if it's in the right position.
  SDValue TCChain = InChain;
  bool isTailCall = isInTailCallPosition(DAG, Op.getNode(), TCChain);
  if (isTailCall)
    InChain = TCChain;

  TargetLowering::
  CallLoweringInfo CLI(InChain, RetTy, false, false, false, false,
                    0, getLibcallCallingConv(Call), isTailCall,
                    /*doesNotReturn=*/false, /*isReturnValueUsed=*/true,
                    Callee, Args, DAG, SDLoc(Op));
  std::pair<SDValue, SDValue> CallInfo = LowerCallTo(CLI);

  if (!CallInfo.second.getNode())
    // It's a tailcall, return the chain (which is the DAG root).
    return DAG.getRoot();

  return CallInfo.first;
}

SDValue
AArch64TargetLowering::LowerFP_ROUND(SDValue Op, SelectionDAG &DAG) const {
  if (Op.getOperand(0).getValueType() != MVT::f128) {
    // It's legal except when f128 is involved
    return Op;
  }

  RTLIB::Libcall LC;
  LC  = RTLIB::getFPROUND(Op.getOperand(0).getValueType(), Op.getValueType());

  SDValue SrcVal = Op.getOperand(0);
  return makeLibCall(DAG, LC, Op.getValueType(), &SrcVal, 1,
                     /*isSigned*/ false, SDLoc(Op));
}

SDValue
AArch64TargetLowering::LowerFP_EXTEND(SDValue Op, SelectionDAG &DAG) const {
  assert(Op.getValueType() == MVT::f128 && "Unexpected lowering");

  RTLIB::Libcall LC;
  LC  = RTLIB::getFPEXT(Op.getOperand(0).getValueType(), Op.getValueType());

  return LowerF128ToCall(Op, DAG, LC);
}

SDValue
AArch64TargetLowering::LowerFP_TO_INT(SDValue Op, SelectionDAG &DAG,
                                      bool IsSigned) const {
  if (Op.getOperand(0).getValueType() != MVT::f128) {
    // It's legal except when f128 is involved
    return Op;
  }

  RTLIB::Libcall LC;
  if (IsSigned)
    LC = RTLIB::getFPTOSINT(Op.getOperand(0).getValueType(), Op.getValueType());
  else
    LC = RTLIB::getFPTOUINT(Op.getOperand(0).getValueType(), Op.getValueType());

  return LowerF128ToCall(Op, DAG, LC);
}

SDValue
AArch64TargetLowering::LowerGlobalAddressELFLarge(SDValue Op,
                                                  SelectionDAG &DAG) const {
  assert(getTargetMachine().getCodeModel() == CodeModel::Large);
  assert(getTargetMachine().getRelocationModel() == Reloc::Static);

  EVT PtrVT = getPointerTy();
  SDLoc dl(Op);
  const GlobalAddressSDNode *GN = cast<GlobalAddressSDNode>(Op);
  const GlobalValue *GV = GN->getGlobal();

  SDValue GlobalAddr = DAG.getNode(
      AArch64ISD::WrapperLarge, dl, PtrVT,
      DAG.getTargetGlobalAddress(GV, dl, PtrVT, 0, AArch64II::MO_ABS_G3),
      DAG.getTargetGlobalAddress(GV, dl, PtrVT, 0, AArch64II::MO_ABS_G2_NC),
      DAG.getTargetGlobalAddress(GV, dl, PtrVT, 0, AArch64II::MO_ABS_G1_NC),
      DAG.getTargetGlobalAddress(GV, dl, PtrVT, 0, AArch64II::MO_ABS_G0_NC));

  if (GN->getOffset() != 0)
    return DAG.getNode(ISD::ADD, dl, PtrVT, GlobalAddr,
                       DAG.getConstant(GN->getOffset(), PtrVT));

  return GlobalAddr;
}

SDValue
AArch64TargetLowering::LowerGlobalAddressELFSmall(SDValue Op,
                                                  SelectionDAG &DAG) const {
  assert(getTargetMachine().getCodeModel() == CodeModel::Small);

  EVT PtrVT = getPointerTy();
  SDLoc dl(Op);
  const GlobalAddressSDNode *GN = cast<GlobalAddressSDNode>(Op);
  const GlobalValue *GV = GN->getGlobal();
  unsigned Alignment = GV->getAlignment();
  Reloc::Model RelocM = getTargetMachine().getRelocationModel();
  if (GV->isWeakForLinker() && GV->isDeclaration() && RelocM == Reloc::Static) {
    // Weak undefined symbols can't use ADRP/ADD pair since they should evaluate
    // to zero when they remain undefined. In PIC mode the GOT can take care of
    // this, but in absolute mode we use a constant pool load.
    SDValue PoolAddr;
    PoolAddr = DAG.getNode(AArch64ISD::WrapperSmall, dl, PtrVT,
                           DAG.getTargetConstantPool(GV, PtrVT, 0, 0,
                                                     AArch64II::MO_NO_FLAG),
                           DAG.getTargetConstantPool(GV, PtrVT, 0, 0,
                                                     AArch64II::MO_LO12),
                           DAG.getConstant(8, MVT::i32));
    SDValue GlobalAddr = DAG.getLoad(PtrVT, dl, DAG.getEntryNode(), PoolAddr,
                                     MachinePointerInfo::getConstantPool(),
                                     /*isVolatile=*/ false,
                                     /*isNonTemporal=*/ true,
                                     /*isInvariant=*/ true, 8);
    if (GN->getOffset() != 0)
      return DAG.getNode(ISD::ADD, dl, PtrVT, GlobalAddr,
                         DAG.getConstant(GN->getOffset(), PtrVT));

    return GlobalAddr;
  }

  if (Alignment == 0) {
    const PointerType *GVPtrTy = cast<PointerType>(GV->getType());
    if (GVPtrTy->getElementType()->isSized()) {
      Alignment
        = getDataLayout()->getABITypeAlignment(GVPtrTy->getElementType());
    } else {
      // Be conservative if we can't guess, not that it really matters:
      // functions and labels aren't valid for loads, and the methods used to
      // actually calculate an address work with any alignment.
      Alignment = 1;
    }
  }

  unsigned char HiFixup, LoFixup;
  bool UseGOT = getSubtarget()->GVIsIndirectSymbol(GV, RelocM);

  if (UseGOT) {
    HiFixup = AArch64II::MO_GOT;
    LoFixup = AArch64II::MO_GOT_LO12;
    Alignment = 8;
  } else {
    HiFixup = AArch64II::MO_NO_FLAG;
    LoFixup = AArch64II::MO_LO12;
  }

  // AArch64's small model demands the following sequence:
  // ADRP x0, somewhere
  // ADD x0, x0, #:lo12:somewhere ; (or LDR directly).
  SDValue GlobalRef = DAG.getNode(AArch64ISD::WrapperSmall, dl, PtrVT,
                                  DAG.getTargetGlobalAddress(GV, dl, PtrVT, 0,
                                                             HiFixup),
                                  DAG.getTargetGlobalAddress(GV, dl, PtrVT, 0,
                                                             LoFixup),
                                  DAG.getConstant(Alignment, MVT::i32));

  if (UseGOT) {
    GlobalRef = DAG.getNode(AArch64ISD::GOTLoad, dl, PtrVT, DAG.getEntryNode(),
                            GlobalRef);
  }

  if (GN->getOffset() != 0)
    return DAG.getNode(ISD::ADD, dl, PtrVT, GlobalRef,
                       DAG.getConstant(GN->getOffset(), PtrVT));

  return GlobalRef;
}

SDValue
AArch64TargetLowering::LowerGlobalAddressELF(SDValue Op,
                                             SelectionDAG &DAG) const {
  // TableGen doesn't have easy access to the CodeModel or RelocationModel, so
  // we make those distinctions here.

  switch (getTargetMachine().getCodeModel()) {
  case CodeModel::Small:
    return LowerGlobalAddressELFSmall(Op, DAG);
  case CodeModel::Large:
    return LowerGlobalAddressELFLarge(Op, DAG);
  default:
    llvm_unreachable("Only small and large code models supported now");
  }
}

SDValue AArch64TargetLowering::LowerTLSDescCall(SDValue SymAddr,
                                                SDValue DescAddr,
                                                SDLoc DL,
                                                SelectionDAG &DAG) const {
  EVT PtrVT = getPointerTy();

  // The function we need to call is simply the first entry in the GOT for this
  // descriptor, load it in preparation.
  SDValue Func, Chain;
  Func = DAG.getNode(AArch64ISD::GOTLoad, DL, PtrVT, DAG.getEntryNode(),
                     DescAddr);

  // The function takes only one argument: the address of the descriptor itself
  // in X0.
  SDValue Glue;
  Chain = DAG.getCopyToReg(DAG.getEntryNode(), DL, AArch64::X0, DescAddr, Glue);
  Glue = Chain.getValue(1);

  // Finally, there's a special calling-convention which means that the lookup
  // must preserve all registers (except X0, obviously).
  const TargetRegisterInfo *TRI  = getTargetMachine().getRegisterInfo();
  const AArch64RegisterInfo *A64RI
    = static_cast<const AArch64RegisterInfo *>(TRI);
  const uint32_t *Mask = A64RI->getTLSDescCallPreservedMask();

  // We're now ready to populate the argument list, as with a normal call:
  std::vector<SDValue> Ops;
  Ops.push_back(Chain);
  Ops.push_back(Func);
  Ops.push_back(SymAddr);
  Ops.push_back(DAG.getRegister(AArch64::X0, PtrVT));
  Ops.push_back(DAG.getRegisterMask(Mask));
  Ops.push_back(Glue);

  SDVTList NodeTys = DAG.getVTList(MVT::Other, MVT::Glue);
  Chain = DAG.getNode(AArch64ISD::TLSDESCCALL, DL, NodeTys, &Ops[0],
                      Ops.size());
  Glue = Chain.getValue(1);

  // After the call, the offset from TPIDR_EL0 is in X0, copy it out and pass it
  // back to the generic handling code.
  return DAG.getCopyFromReg(Chain, DL, AArch64::X0, PtrVT, Glue);
}

SDValue
AArch64TargetLowering::LowerGlobalTLSAddress(SDValue Op,
                                             SelectionDAG &DAG) const {
  assert(getSubtarget()->isTargetELF() &&
         "TLS not implemented for non-ELF targets");
  assert(getTargetMachine().getCodeModel() == CodeModel::Small
         && "TLS only supported in small memory model");
  const GlobalAddressSDNode *GA = cast<GlobalAddressSDNode>(Op);

  TLSModel::Model Model = getTargetMachine().getTLSModel(GA->getGlobal());

  SDValue TPOff;
  EVT PtrVT = getPointerTy();
  SDLoc DL(Op);
  const GlobalValue *GV = GA->getGlobal();

  SDValue ThreadBase = DAG.getNode(AArch64ISD::THREAD_POINTER, DL, PtrVT);

  if (Model == TLSModel::InitialExec) {
    TPOff = DAG.getNode(AArch64ISD::WrapperSmall, DL, PtrVT,
                        DAG.getTargetGlobalAddress(GV, DL, PtrVT, 0,
                                                   AArch64II::MO_GOTTPREL),
                        DAG.getTargetGlobalAddress(GV, DL, PtrVT, 0,
                                                   AArch64II::MO_GOTTPREL_LO12),
                        DAG.getConstant(8, MVT::i32));
    TPOff = DAG.getNode(AArch64ISD::GOTLoad, DL, PtrVT, DAG.getEntryNode(),
                        TPOff);
  } else if (Model == TLSModel::LocalExec) {
    SDValue HiVar = DAG.getTargetGlobalAddress(GV, DL, MVT::i64, 0,
                                               AArch64II::MO_TPREL_G1);
    SDValue LoVar = DAG.getTargetGlobalAddress(GV, DL, MVT::i64, 0,
                                               AArch64II::MO_TPREL_G0_NC);

    TPOff = SDValue(DAG.getMachineNode(AArch64::MOVZxii, DL, PtrVT, HiVar,
                                       DAG.getTargetConstant(0, MVT::i32)), 0);
    TPOff = SDValue(DAG.getMachineNode(AArch64::MOVKxii, DL, PtrVT,
                                       TPOff, LoVar,
                                       DAG.getTargetConstant(0, MVT::i32)), 0);
  } else if (Model == TLSModel::GeneralDynamic) {
    // Accesses used in this sequence go via the TLS descriptor which lives in
    // the GOT. Prepare an address we can use to handle this.
    SDValue HiDesc = DAG.getTargetGlobalAddress(GV, DL, PtrVT, 0,
                                                AArch64II::MO_TLSDESC);
    SDValue LoDesc = DAG.getTargetGlobalAddress(GV, DL, PtrVT, 0,
                                                AArch64II::MO_TLSDESC_LO12);
    SDValue DescAddr = DAG.getNode(AArch64ISD::WrapperSmall, DL, PtrVT,
                                   HiDesc, LoDesc,
                                   DAG.getConstant(8, MVT::i32));
    SDValue SymAddr = DAG.getTargetGlobalAddress(GV, DL, PtrVT, 0);

    TPOff = LowerTLSDescCall(SymAddr, DescAddr, DL, DAG);
  } else if (Model == TLSModel::LocalDynamic) {
    // Local-dynamic accesses proceed in two phases. A general-dynamic TLS
    // descriptor call against the special symbol _TLS_MODULE_BASE_ to calculate
    // the beginning of the module's TLS region, followed by a DTPREL offset
    // calculation.

    // These accesses will need deduplicating if there's more than one.
    AArch64MachineFunctionInfo* MFI = DAG.getMachineFunction()
      .getInfo<AArch64MachineFunctionInfo>();
    MFI->incNumLocalDynamicTLSAccesses();


    // Get the location of _TLS_MODULE_BASE_:
    SDValue HiDesc = DAG.getTargetExternalSymbol("_TLS_MODULE_BASE_", PtrVT,
                                                AArch64II::MO_TLSDESC);
    SDValue LoDesc = DAG.getTargetExternalSymbol("_TLS_MODULE_BASE_", PtrVT,
                                                AArch64II::MO_TLSDESC_LO12);
    SDValue DescAddr = DAG.getNode(AArch64ISD::WrapperSmall, DL, PtrVT,
                                   HiDesc, LoDesc,
                                   DAG.getConstant(8, MVT::i32));
    SDValue SymAddr = DAG.getTargetExternalSymbol("_TLS_MODULE_BASE_", PtrVT);

    ThreadBase = LowerTLSDescCall(SymAddr, DescAddr, DL, DAG);

    // Get the variable's offset from _TLS_MODULE_BASE_
    SDValue HiVar = DAG.getTargetGlobalAddress(GV, DL, MVT::i64, 0,
                                               AArch64II::MO_DTPREL_G1);
    SDValue LoVar = DAG.getTargetGlobalAddress(GV, DL, MVT::i64, 0,
                                               AArch64II::MO_DTPREL_G0_NC);

    TPOff = SDValue(DAG.getMachineNode(AArch64::MOVZxii, DL, PtrVT, HiVar,
                                       DAG.getTargetConstant(0, MVT::i32)), 0);
    TPOff = SDValue(DAG.getMachineNode(AArch64::MOVKxii, DL, PtrVT,
                                       TPOff, LoVar,
                                       DAG.getTargetConstant(0, MVT::i32)), 0);
  } else
      llvm_unreachable("Unsupported TLS access model");


  return DAG.getNode(ISD::ADD, DL, PtrVT, ThreadBase, TPOff);
}

SDValue
AArch64TargetLowering::LowerINT_TO_FP(SDValue Op, SelectionDAG &DAG,
                                      bool IsSigned) const {
  if (Op.getValueType() != MVT::f128) {
    // Legal for everything except f128.
    return Op;
  }

  RTLIB::Libcall LC;
  if (IsSigned)
    LC = RTLIB::getSINTTOFP(Op.getOperand(0).getValueType(), Op.getValueType());
  else
    LC = RTLIB::getUINTTOFP(Op.getOperand(0).getValueType(), Op.getValueType());

  return LowerF128ToCall(Op, DAG, LC);
}


SDValue
AArch64TargetLowering::LowerJumpTable(SDValue Op, SelectionDAG &DAG) const {
  JumpTableSDNode *JT = cast<JumpTableSDNode>(Op);
  SDLoc dl(JT);
  EVT PtrVT = getPointerTy();

  // When compiling PIC, jump tables get put in the code section so a static
  // relocation-style is acceptable for both cases.
  switch (getTargetMachine().getCodeModel()) {
  case CodeModel::Small:
    return DAG.getNode(AArch64ISD::WrapperSmall, dl, PtrVT,
                       DAG.getTargetJumpTable(JT->getIndex(), PtrVT),
                       DAG.getTargetJumpTable(JT->getIndex(), PtrVT,
                                              AArch64II::MO_LO12),
                       DAG.getConstant(1, MVT::i32));
  case CodeModel::Large:
    return DAG.getNode(
      AArch64ISD::WrapperLarge, dl, PtrVT,
      DAG.getTargetJumpTable(JT->getIndex(), PtrVT, AArch64II::MO_ABS_G3),
      DAG.getTargetJumpTable(JT->getIndex(), PtrVT, AArch64II::MO_ABS_G2_NC),
      DAG.getTargetJumpTable(JT->getIndex(), PtrVT, AArch64II::MO_ABS_G1_NC),
      DAG.getTargetJumpTable(JT->getIndex(), PtrVT, AArch64II::MO_ABS_G0_NC));
  default:
    llvm_unreachable("Only small and large code models supported now");
  }
}

// (SELECT_CC lhs, rhs, iftrue, iffalse, condcode)
SDValue
AArch64TargetLowering::LowerSELECT_CC(SDValue Op, SelectionDAG &DAG) const {
  SDLoc dl(Op);
  SDValue LHS = Op.getOperand(0);
  SDValue RHS = Op.getOperand(1);
  SDValue IfTrue = Op.getOperand(2);
  SDValue IfFalse = Op.getOperand(3);
  ISD::CondCode CC = cast<CondCodeSDNode>(Op.getOperand(4))->get();

  if (LHS.getValueType() == MVT::f128) {
    // f128 comparisons are lowered to libcalls, but slot in nicely here
    // afterwards.
    softenSetCCOperands(DAG, MVT::f128, LHS, RHS, CC, dl);

    // If softenSetCCOperands returned a scalar, we need to compare the result
    // against zero to select between true and false values.
    if (RHS.getNode() == 0) {
      RHS = DAG.getConstant(0, LHS.getValueType());
      CC = ISD::SETNE;
    }
  }

  if (LHS.getValueType().isInteger()) {
    SDValue A64cc;

    // Integers are handled in a separate function because the combinations of
    // immediates and tests can get hairy and we may want to fiddle things.
    SDValue CmpOp = getSelectableIntSetCC(LHS, RHS, CC, A64cc, DAG, dl);

    return DAG.getNode(AArch64ISD::SELECT_CC, dl, Op.getValueType(),
                       CmpOp, IfTrue, IfFalse, A64cc);
  }

  // Note that some LLVM floating-point CondCodes can't be lowered to a single
  // conditional branch, hence FPCCToA64CC can set a second test, where either
  // passing is sufficient.
  A64CC::CondCodes CondCode, Alternative = A64CC::Invalid;
  CondCode = FPCCToA64CC(CC, Alternative);
  SDValue A64cc = DAG.getConstant(CondCode, MVT::i32);
  SDValue SetCC = DAG.getNode(AArch64ISD::SETCC, dl, MVT::i32, LHS, RHS,
                              DAG.getCondCode(CC));
  SDValue A64SELECT_CC = DAG.getNode(AArch64ISD::SELECT_CC, dl,
                                     Op.getValueType(),
                                     SetCC, IfTrue, IfFalse, A64cc);

  if (Alternative != A64CC::Invalid) {
    A64cc = DAG.getConstant(Alternative, MVT::i32);
    A64SELECT_CC = DAG.getNode(AArch64ISD::SELECT_CC, dl, Op.getValueType(),
                               SetCC, IfTrue, A64SELECT_CC, A64cc);

  }

  return A64SELECT_CC;
}

// (SELECT testbit, iftrue, iffalse)
SDValue
AArch64TargetLowering::LowerSELECT(SDValue Op, SelectionDAG &DAG) const {
  SDLoc dl(Op);
  SDValue TheBit = Op.getOperand(0);
  SDValue IfTrue = Op.getOperand(1);
  SDValue IfFalse = Op.getOperand(2);

  // AArch64 BooleanContents is the default UndefinedBooleanContent, which means
  // that as the consumer we are responsible for ignoring rubbish in higher
  // bits.
  TheBit = DAG.getNode(ISD::AND, dl, MVT::i32, TheBit,
                       DAG.getConstant(1, MVT::i32));
  SDValue A64CMP = DAG.getNode(AArch64ISD::SETCC, dl, MVT::i32, TheBit,
                               DAG.getConstant(0, TheBit.getValueType()),
                               DAG.getCondCode(ISD::SETNE));

  return DAG.getNode(AArch64ISD::SELECT_CC, dl, Op.getValueType(),
                     A64CMP, IfTrue, IfFalse,
                     DAG.getConstant(A64CC::NE, MVT::i32));
}

// (SETCC lhs, rhs, condcode)
SDValue
AArch64TargetLowering::LowerSETCC(SDValue Op, SelectionDAG &DAG) const {
  SDLoc dl(Op);
  SDValue LHS = Op.getOperand(0);
  SDValue RHS = Op.getOperand(1);
  ISD::CondCode CC = cast<CondCodeSDNode>(Op.getOperand(2))->get();
  EVT VT = Op.getValueType();

  if (LHS.getValueType() == MVT::f128) {
    // f128 comparisons will be lowered to libcalls giving a valid LHS and RHS
    // for the rest of the function (some i32 or i64 values).
    softenSetCCOperands(DAG, MVT::f128, LHS, RHS, CC, dl);

    // If softenSetCCOperands returned a scalar, use it.
    if (RHS.getNode() == 0) {
      assert(LHS.getValueType() == Op.getValueType() &&
             "Unexpected setcc expansion!");
      return LHS;
    }
  }

  if (LHS.getValueType().isInteger()) {
    SDValue A64cc;

    // Integers are handled in a separate function because the combinations of
    // immediates and tests can get hairy and we may want to fiddle things.
    SDValue CmpOp = getSelectableIntSetCC(LHS, RHS, CC, A64cc, DAG, dl);

    return DAG.getNode(AArch64ISD::SELECT_CC, dl, VT,
                       CmpOp, DAG.getConstant(1, VT), DAG.getConstant(0, VT),
                       A64cc);
  }

  // Note that some LLVM floating-point CondCodes can't be lowered to a single
  // conditional branch, hence FPCCToA64CC can set a second test, where either
  // passing is sufficient.
  A64CC::CondCodes CondCode, Alternative = A64CC::Invalid;
  CondCode = FPCCToA64CC(CC, Alternative);
  SDValue A64cc = DAG.getConstant(CondCode, MVT::i32);
  SDValue CmpOp = DAG.getNode(AArch64ISD::SETCC, dl, MVT::i32, LHS, RHS,
                              DAG.getCondCode(CC));
  SDValue A64SELECT_CC = DAG.getNode(AArch64ISD::SELECT_CC, dl, VT,
                                     CmpOp, DAG.getConstant(1, VT),
                                     DAG.getConstant(0, VT), A64cc);

  if (Alternative != A64CC::Invalid) {
    A64cc = DAG.getConstant(Alternative, MVT::i32);
    A64SELECT_CC = DAG.getNode(AArch64ISD::SELECT_CC, dl, VT, CmpOp,
                               DAG.getConstant(1, VT), A64SELECT_CC, A64cc);
  }

  return A64SELECT_CC;
}

SDValue
AArch64TargetLowering::LowerVACOPY(SDValue Op, SelectionDAG &DAG) const {
  const Value *DestSV = cast<SrcValueSDNode>(Op.getOperand(3))->getValue();
  const Value *SrcSV = cast<SrcValueSDNode>(Op.getOperand(3))->getValue();

  // We have to make sure we copy the entire structure: 8+8+8+4+4 = 32 bytes
  // rather than just 8.
  return DAG.getMemcpy(Op.getOperand(0), SDLoc(Op),
                       Op.getOperand(1), Op.getOperand(2),
                       DAG.getConstant(32, MVT::i32), 8, false, false,
                       MachinePointerInfo(DestSV), MachinePointerInfo(SrcSV));
}

SDValue
AArch64TargetLowering::LowerVASTART(SDValue Op, SelectionDAG &DAG) const {
  // The layout of the va_list struct is specified in the AArch64 Procedure Call
  // Standard, section B.3.
  MachineFunction &MF = DAG.getMachineFunction();
  AArch64MachineFunctionInfo *FuncInfo
    = MF.getInfo<AArch64MachineFunctionInfo>();
  SDLoc DL(Op);

  SDValue Chain = Op.getOperand(0);
  SDValue VAList = Op.getOperand(1);
  const Value *SV = cast<SrcValueSDNode>(Op.getOperand(2))->getValue();
  SmallVector<SDValue, 4> MemOps;

  // void *__stack at offset 0
  SDValue Stack = DAG.getFrameIndex(FuncInfo->getVariadicStackIdx(),
                                    getPointerTy());
  MemOps.push_back(DAG.getStore(Chain, DL, Stack, VAList,
                                MachinePointerInfo(SV), false, false, 0));

  // void *__gr_top at offset 8
  int GPRSize = FuncInfo->getVariadicGPRSize();
  if (GPRSize > 0) {
    SDValue GRTop, GRTopAddr;

    GRTopAddr = DAG.getNode(ISD::ADD, DL, getPointerTy(), VAList,
                            DAG.getConstant(8, getPointerTy()));

    GRTop = DAG.getFrameIndex(FuncInfo->getVariadicGPRIdx(), getPointerTy());
    GRTop = DAG.getNode(ISD::ADD, DL, getPointerTy(), GRTop,
                        DAG.getConstant(GPRSize, getPointerTy()));

    MemOps.push_back(DAG.getStore(Chain, DL, GRTop, GRTopAddr,
                                  MachinePointerInfo(SV, 8),
                                  false, false, 0));
  }

  // void *__vr_top at offset 16
  int FPRSize = FuncInfo->getVariadicFPRSize();
  if (FPRSize > 0) {
    SDValue VRTop, VRTopAddr;
    VRTopAddr = DAG.getNode(ISD::ADD, DL, getPointerTy(), VAList,
                            DAG.getConstant(16, getPointerTy()));

    VRTop = DAG.getFrameIndex(FuncInfo->getVariadicFPRIdx(), getPointerTy());
    VRTop = DAG.getNode(ISD::ADD, DL, getPointerTy(), VRTop,
                        DAG.getConstant(FPRSize, getPointerTy()));

    MemOps.push_back(DAG.getStore(Chain, DL, VRTop, VRTopAddr,
                                  MachinePointerInfo(SV, 16),
                                  false, false, 0));
  }

  // int __gr_offs at offset 24
  SDValue GROffsAddr = DAG.getNode(ISD::ADD, DL, getPointerTy(), VAList,
                                   DAG.getConstant(24, getPointerTy()));
  MemOps.push_back(DAG.getStore(Chain, DL, DAG.getConstant(-GPRSize, MVT::i32),
                                GROffsAddr, MachinePointerInfo(SV, 24),
                                false, false, 0));

  // int __vr_offs at offset 28
  SDValue VROffsAddr = DAG.getNode(ISD::ADD, DL, getPointerTy(), VAList,
                                   DAG.getConstant(28, getPointerTy()));
  MemOps.push_back(DAG.getStore(Chain, DL, DAG.getConstant(-FPRSize, MVT::i32),
                                VROffsAddr, MachinePointerInfo(SV, 28),
                                false, false, 0));

  return DAG.getNode(ISD::TokenFactor, DL, MVT::Other, &MemOps[0],
                     MemOps.size());
}

SDValue
AArch64TargetLowering::LowerOperation(SDValue Op, SelectionDAG &DAG) const {
  switch (Op.getOpcode()) {
  default: llvm_unreachable("Don't know how to custom lower this!");
  case ISD::FADD: return LowerF128ToCall(Op, DAG, RTLIB::ADD_F128);
  case ISD::FSUB: return LowerF128ToCall(Op, DAG, RTLIB::SUB_F128);
  case ISD::FMUL: return LowerF128ToCall(Op, DAG, RTLIB::MUL_F128);
  case ISD::FDIV: return LowerF128ToCall(Op, DAG, RTLIB::DIV_F128);
  case ISD::FP_TO_SINT: return LowerFP_TO_INT(Op, DAG, true);
  case ISD::FP_TO_UINT: return LowerFP_TO_INT(Op, DAG, false);
  case ISD::SINT_TO_FP: return LowerINT_TO_FP(Op, DAG, true);
  case ISD::UINT_TO_FP: return LowerINT_TO_FP(Op, DAG, false);
  case ISD::FP_ROUND: return LowerFP_ROUND(Op, DAG);
  case ISD::FP_EXTEND: return LowerFP_EXTEND(Op, DAG);

  case ISD::BlockAddress: return LowerBlockAddress(Op, DAG);
  case ISD::BRCOND: return LowerBRCOND(Op, DAG);
  case ISD::BR_CC: return LowerBR_CC(Op, DAG);
  case ISD::GlobalAddress: return LowerGlobalAddressELF(Op, DAG);
  case ISD::GlobalTLSAddress: return LowerGlobalTLSAddress(Op, DAG);
  case ISD::JumpTable: return LowerJumpTable(Op, DAG);
  case ISD::SELECT: return LowerSELECT(Op, DAG);
  case ISD::SELECT_CC: return LowerSELECT_CC(Op, DAG);
  case ISD::SETCC: return LowerSETCC(Op, DAG);
  case ISD::VACOPY: return LowerVACOPY(Op, DAG);
  case ISD::VASTART: return LowerVASTART(Op, DAG);
  }

  return SDValue();
}

static SDValue PerformANDCombine(SDNode *N,
                                 TargetLowering::DAGCombinerInfo &DCI) {

  SelectionDAG &DAG = DCI.DAG;
  SDLoc DL(N);
  EVT VT = N->getValueType(0);

  // We're looking for an SRA/SHL pair which form an SBFX.

  if (VT != MVT::i32 && VT != MVT::i64)
    return SDValue();

  if (!isa<ConstantSDNode>(N->getOperand(1)))
    return SDValue();

  uint64_t TruncMask = N->getConstantOperandVal(1);
  if (!isMask_64(TruncMask))
    return SDValue();

  uint64_t Width = CountPopulation_64(TruncMask);
  SDValue Shift = N->getOperand(0);

  if (Shift.getOpcode() != ISD::SRL)
    return SDValue();

  if (!isa<ConstantSDNode>(Shift->getOperand(1)))
    return SDValue();
  uint64_t LSB = Shift->getConstantOperandVal(1);

  if (LSB > VT.getSizeInBits() || Width > VT.getSizeInBits())
    return SDValue();

  return DAG.getNode(AArch64ISD::UBFX, DL, VT, Shift.getOperand(0),
                     DAG.getConstant(LSB, MVT::i64),
                     DAG.getConstant(LSB + Width - 1, MVT::i64));
}

/// For a true bitfield insert, the bits getting into that contiguous mask
/// should come from the low part of an existing value: they must be formed from
/// a compatible SHL operation (unless they're already low). This function
/// checks that condition and returns the least-significant bit that's
/// intended. If the operation not a field preparation, -1 is returned.
static int32_t getLSBForBFI(SelectionDAG &DAG, SDLoc DL, EVT VT,
                            SDValue &MaskedVal, uint64_t Mask) {
  if (!isShiftedMask_64(Mask))
    return -1;

  // Now we need to alter MaskedVal so that it is an appropriate input for a BFI
  // instruction. BFI will do a left-shift by LSB before applying the mask we've
  // spotted, so in general we should pre-emptively "undo" that by making sure
  // the incoming bits have had a right-shift applied to them.
  //
  // This right shift, however, will combine with existing left/right shifts. In
  // the simplest case of a completely straight bitfield operation, it will be
  // expected to completely cancel out with an existing SHL. More complicated
  // cases (e.g. bitfield to bitfield copy) may still need a real shift before
  // the BFI.

  uint64_t LSB = countTrailingZeros(Mask);
  int64_t ShiftRightRequired = LSB;
  if (MaskedVal.getOpcode() == ISD::SHL &&
      isa<ConstantSDNode>(MaskedVal.getOperand(1))) {
    ShiftRightRequired -= MaskedVal.getConstantOperandVal(1);
    MaskedVal = MaskedVal.getOperand(0);
  } else if (MaskedVal.getOpcode() == ISD::SRL &&
             isa<ConstantSDNode>(MaskedVal.getOperand(1))) {
    ShiftRightRequired += MaskedVal.getConstantOperandVal(1);
    MaskedVal = MaskedVal.getOperand(0);
  }

  if (ShiftRightRequired > 0)
    MaskedVal = DAG.getNode(ISD::SRL, DL, VT, MaskedVal,
                            DAG.getConstant(ShiftRightRequired, MVT::i64));
  else if (ShiftRightRequired < 0) {
    // We could actually end up with a residual left shift, for example with
    // "struc.bitfield = val << 1".
    MaskedVal = DAG.getNode(ISD::SHL, DL, VT, MaskedVal,
                            DAG.getConstant(-ShiftRightRequired, MVT::i64));
  }

  return LSB;
}

/// Searches from N for an existing AArch64ISD::BFI node, possibly surrounded by
/// a mask and an extension. Returns true if a BFI was found and provides
/// information on its surroundings.
static bool findMaskedBFI(SDValue N, SDValue &BFI, uint64_t &Mask,
                          bool &Extended) {
  Extended = false;
  if (N.getOpcode() == ISD::ZERO_EXTEND) {
    Extended = true;
    N = N.getOperand(0);
  }

  if (N.getOpcode() == ISD::AND && isa<ConstantSDNode>(N.getOperand(1))) {
    Mask = N->getConstantOperandVal(1);
    N = N.getOperand(0);
  } else {
    // Mask is the whole width.
    Mask = -1ULL >> (64 - N.getValueType().getSizeInBits());
  }

  if (N.getOpcode() == AArch64ISD::BFI) {
    BFI = N;
    return true;
  }

  return false;
}

/// Try to combine a subtree (rooted at an OR) into a "masked BFI" node, which
/// is roughly equivalent to (and (BFI ...), mask). This form is used because it
/// can often be further combined with a larger mask. Ultimately, we want mask
/// to be 2^32-1 or 2^64-1 so the AND can be skipped.
static SDValue tryCombineToBFI(SDNode *N,
                               TargetLowering::DAGCombinerInfo &DCI,
                               const AArch64Subtarget *Subtarget) {
  SelectionDAG &DAG = DCI.DAG;
  SDLoc DL(N);
  EVT VT = N->getValueType(0);

  assert(N->getOpcode() == ISD::OR && "Unexpected root");

  // We need the LHS to be (and SOMETHING, MASK). Find out what that mask is or
  // abandon the effort.
  SDValue LHS = N->getOperand(0);
  if (LHS.getOpcode() != ISD::AND)
    return SDValue();

  uint64_t LHSMask;
  if (isa<ConstantSDNode>(LHS.getOperand(1)))
    LHSMask = LHS->getConstantOperandVal(1);
  else
    return SDValue();

  // We also need the RHS to be (and SOMETHING, MASK). Find out what that mask
  // is or abandon the effort.
  SDValue RHS = N->getOperand(1);
  if (RHS.getOpcode() != ISD::AND)
    return SDValue();

  uint64_t RHSMask;
  if (isa<ConstantSDNode>(RHS.getOperand(1)))
    RHSMask = RHS->getConstantOperandVal(1);
  else
    return SDValue();

  // Can't do anything if the masks are incompatible.
  if (LHSMask & RHSMask)
    return SDValue();

  // Now we need one of the masks to be a contiguous field. Without loss of
  // generality that should be the RHS one.
  SDValue Bitfield = LHS.getOperand(0);
  if (getLSBForBFI(DAG, DL, VT, Bitfield, LHSMask) != -1) {
    // We know that LHS is a candidate new value, and RHS isn't already a better
    // one.
    std::swap(LHS, RHS);
    std::swap(LHSMask, RHSMask);
  }

  // We've done our best to put the right operands in the right places, all we
  // can do now is check whether a BFI exists.
  Bitfield = RHS.getOperand(0);
  int32_t LSB = getLSBForBFI(DAG, DL, VT, Bitfield, RHSMask);
  if (LSB == -1)
    return SDValue();

  uint32_t Width = CountPopulation_64(RHSMask);
  assert(Width && "Expected non-zero bitfield width");

  SDValue BFI = DAG.getNode(AArch64ISD::BFI, DL, VT,
                            LHS.getOperand(0), Bitfield,
                            DAG.getConstant(LSB, MVT::i64),
                            DAG.getConstant(Width, MVT::i64));

  // Mask is trivial
  if ((LHSMask | RHSMask) == (-1ULL >> (64 - VT.getSizeInBits())))
    return BFI;

  return DAG.getNode(ISD::AND, DL, VT, BFI,
                     DAG.getConstant(LHSMask | RHSMask, VT));
}

/// Search for the bitwise combining (with careful masks) of a MaskedBFI and its
/// original input. This is surprisingly common because SROA splits things up
/// into i8 chunks, so the originally detected MaskedBFI may actually only act
/// on the low (say) byte of a word. This is then orred into the rest of the
/// word afterwards.
///
/// Basic input: (or (and OLDFIELD, MASK1), (MaskedBFI MASK2, OLDFIELD, ...)).
///
/// If MASK1 and MASK2 are compatible, we can fold the whole thing into the
/// MaskedBFI. We can also deal with a certain amount of extend/truncate being
/// involved.
static SDValue tryCombineToLargerBFI(SDNode *N,
                                     TargetLowering::DAGCombinerInfo &DCI,
                                     const AArch64Subtarget *Subtarget) {
  SelectionDAG &DAG = DCI.DAG;
  SDLoc DL(N);
  EVT VT = N->getValueType(0);

  // First job is to hunt for a MaskedBFI on either the left or right. Swap
  // operands if it's actually on the right.
  SDValue BFI;
  SDValue PossExtraMask;
  uint64_t ExistingMask = 0;
  bool Extended = false;
  if (findMaskedBFI(N->getOperand(0), BFI, ExistingMask, Extended))
    PossExtraMask = N->getOperand(1);
  else if (findMaskedBFI(N->getOperand(1), BFI, ExistingMask, Extended))
    PossExtraMask = N->getOperand(0);
  else
    return SDValue();

  // We can only combine a BFI with another compatible mask.
  if (PossExtraMask.getOpcode() != ISD::AND ||
      !isa<ConstantSDNode>(PossExtraMask.getOperand(1)))
    return SDValue();

  uint64_t ExtraMask = PossExtraMask->getConstantOperandVal(1);

  // Masks must be compatible.
  if (ExtraMask & ExistingMask)
    return SDValue();

  SDValue OldBFIVal = BFI.getOperand(0);
  SDValue NewBFIVal = BFI.getOperand(1);
  if (Extended) {
    // We skipped a ZERO_EXTEND above, so the input to the MaskedBFIs should be
    // 32-bit and we'll be forming a 64-bit MaskedBFI. The MaskedBFI arguments
    // need to be made compatible.
    assert(VT == MVT::i64 && BFI.getValueType() == MVT::i32
           && "Invalid types for BFI");
    OldBFIVal = DAG.getNode(ISD::ANY_EXTEND, DL, VT, OldBFIVal);
    NewBFIVal = DAG.getNode(ISD::ANY_EXTEND, DL, VT, NewBFIVal);
  }

  // We need the MaskedBFI to be combined with a mask of the *same* value.
  if (PossExtraMask.getOperand(0) != OldBFIVal)
    return SDValue();

  BFI = DAG.getNode(AArch64ISD::BFI, DL, VT,
                    OldBFIVal, NewBFIVal,
                    BFI.getOperand(2), BFI.getOperand(3));

  // If the masking is trivial, we don't need to create it.
  if ((ExtraMask | ExistingMask) == (-1ULL >> (64 - VT.getSizeInBits())))
    return BFI;

  return DAG.getNode(ISD::AND, DL, VT, BFI,
                     DAG.getConstant(ExtraMask | ExistingMask, VT));
}

/// An EXTR instruction is made up of two shifts, ORed together. This helper
/// searches for and classifies those shifts.
static bool findEXTRHalf(SDValue N, SDValue &Src, uint32_t &ShiftAmount,
                         bool &FromHi) {
  if (N.getOpcode() == ISD::SHL)
    FromHi = false;
  else if (N.getOpcode() == ISD::SRL)
    FromHi = true;
  else
    return false;

  if (!isa<ConstantSDNode>(N.getOperand(1)))
    return false;

  ShiftAmount = N->getConstantOperandVal(1);
  Src = N->getOperand(0);
  return true;
}

/// EXTR instruction extracts a contiguous chunk of bits from two existing
/// registers viewed as a high/low pair. This function looks for the pattern:
/// (or (shl VAL1, #N), (srl VAL2, #RegWidth-N)) and replaces it with an
/// EXTR. Can't quite be done in TableGen because the two immediates aren't
/// independent.
static SDValue tryCombineToEXTR(SDNode *N,
                                TargetLowering::DAGCombinerInfo &DCI) {
  SelectionDAG &DAG = DCI.DAG;
  SDLoc DL(N);
  EVT VT = N->getValueType(0);

  assert(N->getOpcode() == ISD::OR && "Unexpected root");

  if (VT != MVT::i32 && VT != MVT::i64)
    return SDValue();

  SDValue LHS;
  uint32_t ShiftLHS = 0;
  bool LHSFromHi = 0;
  if (!findEXTRHalf(N->getOperand(0), LHS, ShiftLHS, LHSFromHi))
    return SDValue();

  SDValue RHS;
  uint32_t ShiftRHS = 0;
  bool RHSFromHi = 0;
  if (!findEXTRHalf(N->getOperand(1), RHS, ShiftRHS, RHSFromHi))
    return SDValue();

  // If they're both trying to come from the high part of the register, they're
  // not really an EXTR.
  if (LHSFromHi == RHSFromHi)
    return SDValue();

  if (ShiftLHS + ShiftRHS != VT.getSizeInBits())
    return SDValue();

  if (LHSFromHi) {
    std::swap(LHS, RHS);
    std::swap(ShiftLHS, ShiftRHS);
  }

  return DAG.getNode(AArch64ISD::EXTR, DL, VT,
                     LHS, RHS,
                     DAG.getConstant(ShiftRHS, MVT::i64));
}

/// Target-specific dag combine xforms for ISD::OR
static SDValue PerformORCombine(SDNode *N,
                                TargetLowering::DAGCombinerInfo &DCI,
                                const AArch64Subtarget *Subtarget) {

  SelectionDAG &DAG = DCI.DAG;
  EVT VT = N->getValueType(0);

  if(!DAG.getTargetLoweringInfo().isTypeLegal(VT))
    return SDValue();

  // Attempt to recognise bitfield-insert operations.
  SDValue Res = tryCombineToBFI(N, DCI, Subtarget);
  if (Res.getNode())
    return Res;

  // Attempt to combine an existing MaskedBFI operation into one with a larger
  // mask.
  Res = tryCombineToLargerBFI(N, DCI, Subtarget);
  if (Res.getNode())
    return Res;

  Res = tryCombineToEXTR(N, DCI);
  if (Res.getNode())
    return Res;

  return SDValue();
}

/// Target-specific dag combine xforms for ISD::SRA
static SDValue PerformSRACombine(SDNode *N,
                                 TargetLowering::DAGCombinerInfo &DCI) {

  SelectionDAG &DAG = DCI.DAG;
  SDLoc DL(N);
  EVT VT = N->getValueType(0);

  // We're looking for an SRA/SHL pair which form an SBFX.

  if (VT != MVT::i32 && VT != MVT::i64)
    return SDValue();

  if (!isa<ConstantSDNode>(N->getOperand(1)))
    return SDValue();

  uint64_t ExtraSignBits = N->getConstantOperandVal(1);
  SDValue Shift = N->getOperand(0);

  if (Shift.getOpcode() != ISD::SHL)
    return SDValue();

  if (!isa<ConstantSDNode>(Shift->getOperand(1)))
    return SDValue();

  uint64_t BitsOnLeft = Shift->getConstantOperandVal(1);
  uint64_t Width = VT.getSizeInBits() - ExtraSignBits;
  uint64_t LSB = VT.getSizeInBits() - Width - BitsOnLeft;

  if (LSB > VT.getSizeInBits() || Width > VT.getSizeInBits())
    return SDValue();

  return DAG.getNode(AArch64ISD::SBFX, DL, VT, Shift.getOperand(0),
                     DAG.getConstant(LSB, MVT::i64),
                     DAG.getConstant(LSB + Width - 1, MVT::i64));
}


SDValue
AArch64TargetLowering::PerformDAGCombine(SDNode *N,
                                         DAGCombinerInfo &DCI) const {
  switch (N->getOpcode()) {
  default: break;
  case ISD::AND: return PerformANDCombine(N, DCI);
  case ISD::OR: return PerformORCombine(N, DCI, getSubtarget());
  case ISD::SRA: return PerformSRACombine(N, DCI);
  }
  return SDValue();
}

bool
AArch64TargetLowering::isFMAFasterThanFMulAndFAdd(EVT VT) const {
  VT = VT.getScalarType();

  if (!VT.isSimple())
    return false;

  switch (VT.getSimpleVT().SimpleTy) {
  case MVT::f16:
  case MVT::f32:
  case MVT::f64:
    return true;
  case MVT::f128:
    return false;
  default:
    break;
  }

  return false;
}

AArch64TargetLowering::ConstraintType
AArch64TargetLowering::getConstraintType(const std::string &Constraint) const {
  if (Constraint.size() == 1) {
    switch (Constraint[0]) {
    default: break;
    case 'w': // An FP/SIMD vector register
      return C_RegisterClass;
    case 'I': // Constant that can be used with an ADD instruction
    case 'J': // Constant that can be used with a SUB instruction
    case 'K': // Constant that can be used with a 32-bit logical instruction
    case 'L': // Constant that can be used with a 64-bit logical instruction
    case 'M': // Constant that can be used as a 32-bit MOV immediate
    case 'N': // Constant that can be used as a 64-bit MOV immediate
    case 'Y': // Floating point constant zero
    case 'Z': // Integer constant zero
      return C_Other;
    case 'Q': // A memory reference with base register and no offset
      return C_Memory;
    case 'S': // A symbolic address
      return C_Other;
    }
  }

  // FIXME: Ump, Utf, Usa, Ush
  // Ump: A memory address suitable for ldp/stp in SI, DI, SF and DF modes,
  //      whatever they may be
  // Utf: A memory address suitable for ldp/stp in TF mode, whatever it may be
  // Usa: An absolute symbolic address
  // Ush: The high part (bits 32:12) of a pc-relative symbolic address
  assert(Constraint != "Ump" && Constraint != "Utf" && Constraint != "Usa"
         && Constraint != "Ush" && "Unimplemented constraints");

  return TargetLowering::getConstraintType(Constraint);
}

TargetLowering::ConstraintWeight
AArch64TargetLowering::getSingleConstraintMatchWeight(AsmOperandInfo &Info,
                                                const char *Constraint) const {

  llvm_unreachable("Constraint weight unimplemented");
}

void
AArch64TargetLowering::LowerAsmOperandForConstraint(SDValue Op,
                                                    std::string &Constraint,
                                                    std::vector<SDValue> &Ops,
                                                    SelectionDAG &DAG) const {
  SDValue Result(0, 0);

  // Only length 1 constraints are C_Other.
  if (Constraint.size() != 1) return;

  // Only C_Other constraints get lowered like this. That means constants for us
  // so return early if there's no hope the constraint can be lowered.

  switch(Constraint[0]) {
  default: break;
  case 'I': case 'J': case 'K': case 'L':
  case 'M': case 'N': case 'Z': {
    ConstantSDNode *C = dyn_cast<ConstantSDNode>(Op);
    if (!C)
      return;

    uint64_t CVal = C->getZExtValue();
    uint32_t Bits;

    switch (Constraint[0]) {
    default:
      // FIXME: 'M' and 'N' are MOV pseudo-insts -- unsupported in assembly. 'J'
      // is a peculiarly useless SUB constraint.
      llvm_unreachable("Unimplemented C_Other constraint");
    case 'I':
      if (CVal <= 0xfff)
        break;
      return;
    case 'K':
      if (A64Imms::isLogicalImm(32, CVal, Bits))
        break;
      return;
    case 'L':
      if (A64Imms::isLogicalImm(64, CVal, Bits))
        break;
      return;
    case 'Z':
      if (CVal == 0)
        break;
      return;
    }

    Result = DAG.getTargetConstant(CVal, Op.getValueType());
    break;
  }
  case 'S': {
    // An absolute symbolic address or label reference.
    if (const GlobalAddressSDNode *GA = dyn_cast<GlobalAddressSDNode>(Op)) {
      Result = DAG.getTargetGlobalAddress(GA->getGlobal(), SDLoc(Op),
                                          GA->getValueType(0));
    } else if (const BlockAddressSDNode *BA
                 = dyn_cast<BlockAddressSDNode>(Op)) {
      Result = DAG.getTargetBlockAddress(BA->getBlockAddress(),
                                         BA->getValueType(0));
    } else if (const ExternalSymbolSDNode *ES
                 = dyn_cast<ExternalSymbolSDNode>(Op)) {
      Result = DAG.getTargetExternalSymbol(ES->getSymbol(),
                                           ES->getValueType(0));
    } else
      return;
    break;
  }
  case 'Y':
    if (const ConstantFPSDNode *CFP = dyn_cast<ConstantFPSDNode>(Op)) {
      if (CFP->isExactlyValue(0.0)) {
        Result = DAG.getTargetConstantFP(0.0, CFP->getValueType(0));
        break;
      }
    }
    return;
  }

  if (Result.getNode()) {
    Ops.push_back(Result);
    return;
  }

  // It's an unknown constraint for us. Let generic code have a go.
  TargetLowering::LowerAsmOperandForConstraint(Op, Constraint, Ops, DAG);
}

std::pair<unsigned, const TargetRegisterClass*>
AArch64TargetLowering::getRegForInlineAsmConstraint(
                                                  const std::string &Constraint,
                                                  MVT VT) const {
  if (Constraint.size() == 1) {
    switch (Constraint[0]) {
    case 'r':
      if (VT.getSizeInBits() <= 32)
        return std::make_pair(0U, &AArch64::GPR32RegClass);
      else if (VT == MVT::i64)
        return std::make_pair(0U, &AArch64::GPR64RegClass);
      break;
    case 'w':
      if (VT == MVT::f16)
        return std::make_pair(0U, &AArch64::FPR16RegClass);
      else if (VT == MVT::f32)
        return std::make_pair(0U, &AArch64::FPR32RegClass);
      else if (VT == MVT::f64)
        return std::make_pair(0U, &AArch64::FPR64RegClass);
      else if (VT.getSizeInBits() == 64)
        return std::make_pair(0U, &AArch64::VPR64RegClass);
      else if (VT == MVT::f128)
        return std::make_pair(0U, &AArch64::FPR128RegClass);
      else if (VT.getSizeInBits() == 128)
        return std::make_pair(0U, &AArch64::VPR128RegClass);
      break;
    }
  }

  // Use the default implementation in TargetLowering to convert the register
  // constraint into a member of a register class.
  return TargetLowering::getRegForInlineAsmConstraint(Constraint, VT);
}
