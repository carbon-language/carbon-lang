//===-- CallingConvLower.cpp - Calling Conventions ------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the CCState class, used for lowering and implementing
// calling conventions.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/CallingConvLower.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetRegisterInfo.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Target/TargetMachine.h"
using namespace llvm;

CCState::CCState(CallingConv::ID CC, bool isVarArg, const TargetMachine &tm,
                 SmallVector<CCValAssign, 16> &locs, LLVMContext &C)
  : CallingConv(CC), IsVarArg(isVarArg), TM(tm),
    TRI(*TM.getRegisterInfo()), Locs(locs), Context(C) {
  // No stack is used.
  StackOffset = 0;
  
  UsedRegs.resize((TRI.getNumRegs()+31)/32);
}

// HandleByVal - Allocate a stack slot large enough to pass an argument by
// value. The size and alignment information of the argument is encoded in its
// parameter attribute.
void CCState::HandleByVal(unsigned ValNo, EVT ValVT,
                          EVT LocVT, CCValAssign::LocInfo LocInfo,
                          int MinSize, int MinAlign,
                          ISD::ArgFlagsTy ArgFlags) {
  unsigned Align = ArgFlags.getByValAlign();
  unsigned Size  = ArgFlags.getByValSize();
  if (MinSize > (int)Size)
    Size = MinSize;
  if (MinAlign > (int)Align)
    Align = MinAlign;
  unsigned Offset = AllocateStack(Size, Align);

  addLoc(CCValAssign::getMem(ValNo, ValVT, Offset, LocVT, LocInfo));
}

/// MarkAllocated - Mark a register and all of its aliases as allocated.
void CCState::MarkAllocated(unsigned Reg) {
  UsedRegs[Reg/32] |= 1 << (Reg&31);
  
  if (const unsigned *RegAliases = TRI.getAliasSet(Reg))
    for (; (Reg = *RegAliases); ++RegAliases)
      UsedRegs[Reg/32] |= 1 << (Reg&31);
}

/// AnalyzeFormalArguments - Analyze an array of argument values,
/// incorporating info about the formals into this state.
void
CCState::AnalyzeFormalArguments(const SmallVectorImpl<ISD::InputArg> &Ins,
                                CCAssignFn Fn) {
  unsigned NumArgs = Ins.size();

  for (unsigned i = 0; i != NumArgs; ++i) {
    EVT ArgVT = Ins[i].VT;
    ISD::ArgFlagsTy ArgFlags = Ins[i].Flags;
    if (Fn(i, ArgVT, ArgVT, CCValAssign::Full, ArgFlags, *this)) {
#ifndef NDEBUG
      errs() << "Formal argument #" << i << " has unhandled type "
             << ArgVT.getEVTString();
#endif
      llvm_unreachable(0);
    }
  }
}

/// AnalyzeReturn - Analyze the returned values of a return,
/// incorporating info about the result values into this state.
void CCState::AnalyzeReturn(const SmallVectorImpl<ISD::OutputArg> &Outs,
                            CCAssignFn Fn) {
  // Determine which register each value should be copied into.
  for (unsigned i = 0, e = Outs.size(); i != e; ++i) {
    EVT VT = Outs[i].Val.getValueType();
    ISD::ArgFlagsTy ArgFlags = Outs[i].Flags;
    if (Fn(i, VT, VT, CCValAssign::Full, ArgFlags, *this)) {
#ifndef NDEBUG
      errs() << "Return operand #" << i << " has unhandled type "
             << VT.getEVTString();
#endif
      llvm_unreachable(0);
    }
  }
}


/// AnalyzeCallOperands - Analyze the outgoing arguments to a call,
/// incorporating info about the passed values into this state.
void CCState::AnalyzeCallOperands(const SmallVectorImpl<ISD::OutputArg> &Outs,
                                  CCAssignFn Fn) {
  unsigned NumOps = Outs.size();
  for (unsigned i = 0; i != NumOps; ++i) {
    EVT ArgVT = Outs[i].Val.getValueType();
    ISD::ArgFlagsTy ArgFlags = Outs[i].Flags;
    if (Fn(i, ArgVT, ArgVT, CCValAssign::Full, ArgFlags, *this)) {
#ifndef NDEBUG
      errs() << "Call operand #" << i << " has unhandled type "
             << ArgVT.getEVTString();
#endif
      llvm_unreachable(0);
    }
  }
}

/// AnalyzeCallOperands - Same as above except it takes vectors of types
/// and argument flags.
void CCState::AnalyzeCallOperands(SmallVectorImpl<EVT> &ArgVTs,
                                  SmallVectorImpl<ISD::ArgFlagsTy> &Flags,
                                  CCAssignFn Fn) {
  unsigned NumOps = ArgVTs.size();
  for (unsigned i = 0; i != NumOps; ++i) {
    EVT ArgVT = ArgVTs[i];
    ISD::ArgFlagsTy ArgFlags = Flags[i];
    if (Fn(i, ArgVT, ArgVT, CCValAssign::Full, ArgFlags, *this)) {
#ifndef NDEBUG
      errs() << "Call operand #" << i << " has unhandled type "
             << ArgVT.getEVTString();
#endif
      llvm_unreachable(0);
    }
  }
}

/// AnalyzeCallResult - Analyze the return values of a call,
/// incorporating info about the passed values into this state.
void CCState::AnalyzeCallResult(const SmallVectorImpl<ISD::InputArg> &Ins,
                                CCAssignFn Fn) {
  for (unsigned i = 0, e = Ins.size(); i != e; ++i) {
    EVT VT = Ins[i].VT;
    ISD::ArgFlagsTy Flags = Ins[i].Flags;
    if (Fn(i, VT, VT, CCValAssign::Full, Flags, *this)) {
#ifndef NDEBUG
      errs() << "Call result #" << i << " has unhandled type "
             << VT.getEVTString();
#endif
      llvm_unreachable(0);
    }
  }
}

/// AnalyzeCallResult - Same as above except it's specialized for calls which
/// produce a single value.
void CCState::AnalyzeCallResult(EVT VT, CCAssignFn Fn) {
  if (Fn(0, VT, VT, CCValAssign::Full, ISD::ArgFlagsTy(), *this)) {
#ifndef NDEBUG
    errs() << "Call result has unhandled type "
           << VT.getEVTString();
#endif
    llvm_unreachable(0);
  }
}
