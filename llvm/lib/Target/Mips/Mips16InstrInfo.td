//===- Mips16InstrInfo.td - Target Description for Mips16  -*- tablegen -*-=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file describes Mips16 instructions.
//
//===----------------------------------------------------------------------===//

let isReturn=1, isTerminator=1, hasDelaySlot=1, isCodeGenOnly=1,
    isBarrier=1, hasCtrlDep=1, rx=0b000, ry=0b001 in
def RET16 : FRR16 <0, (outs), (ins CPURAReg:$target),
                   "jr\t$target", [(MipsRet CPURAReg:$target)], IIBranch>,
            Requires<[InMips16Mode]>;
