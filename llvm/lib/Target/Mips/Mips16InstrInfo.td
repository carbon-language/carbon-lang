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
    isBarrier=1, hasCtrlDep=1, rx=0, nd=0, l=0, ra=0  in
def RET16 : FRR16_JALRC < (outs), (ins CPURAReg:$target),
                   "jr\t$target", [(MipsRet CPURAReg:$target)], IIBranch>;

// As stack alignment is always done with addiu, we need a 16-bit immediate
let Defs = [SP], Uses = [SP] in {
def ADJCALLSTACKDOWN16 : MipsPseudo16<(outs), (ins uimm16:$amt),
                                  "!ADJCALLSTACKDOWN $amt",
                                  [(callseq_start timm:$amt)]>;
def ADJCALLSTACKUP16   : MipsPseudo16<(outs), (ins uimm16:$amt1, uimm16:$amt2),
                                  "!ADJCALLSTACKUP $amt1",
                                  [(callseq_end timm:$amt1, timm:$amt2)]>;
}


// Jump and Link (Call)
let isCall=1, hasDelaySlot=1, nd=0, l=0, ra=0 in
def JumpLinkReg16:
    FRR16_JALRC<(outs), (ins CPU16Regs:$rs, variable_ops),
       "jalr \t$rs", [(MipsJmpLink CPU16Regs:$rs)], IIBranch>;
