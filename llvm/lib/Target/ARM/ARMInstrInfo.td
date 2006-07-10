//===- ARMInstrInfo.td - Target Description for ARM Target ----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the "Instituto Nokia de Tecnologia" and
// is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file describes the ARM instructions in TableGen format.
//
//===----------------------------------------------------------------------===//

// Define ARM specific addressing mode.
 //register or frame index
def raddr  : ComplexPattern<iPTR, 1, "SelectAddrReg", []>;

//===----------------------------------------------------------------------===//
// Instructions
//===----------------------------------------------------------------------===//

class InstARM<dag ops, string asmstr, list<dag> pattern> : Instruction {
  let Namespace = "ARM";

  dag OperandList = ops;
  let AsmString   = asmstr;
  let Pattern = pattern;
}

def SDT_ARMCallSeq : SDTypeProfile<0, 1, [ SDTCisVT<0, i32> ]>;
def callseq_start  : SDNode<"ISD::CALLSEQ_START", SDT_ARMCallSeq, [SDNPHasChain]>;
def callseq_end    : SDNode<"ISD::CALLSEQ_END",   SDT_ARMCallSeq, [SDNPHasChain]>;

def ADJCALLSTACKUP : InstARM<(ops i32imm:$amt),
                            "!ADJCALLSTACKUP $amt",
                            [(callseq_end imm:$amt)]>;

def ADJCALLSTACKDOWN : InstARM<(ops i32imm:$amt),
                               "!ADJCALLSTACKDOWN $amt",
                               [(callseq_start imm:$amt)]>;

def bxr: InstARM<(ops IntRegs:$dst), "bx $dst", [(brind IntRegs:$dst)]>;

def ldr   : InstARM<(ops IntRegs:$dst, IntRegs:$addr),
                     "ldr $dst, [$addr]",
                     [(set IntRegs:$dst, (load raddr:$addr))]>;

def str  : InstARM<(ops IntRegs:$src, IntRegs:$addr),
                    "str $src, [$addr]",
                    [(store IntRegs:$src, IntRegs:$addr)]>;

def movrr   : InstARM<(ops IntRegs:$dst, IntRegs:$src),
                       "mov $dst, $src", []>;

def movri   : InstARM<(ops IntRegs:$dst, i32imm:$src),
                       "mov $dst, $src", [(set IntRegs:$dst, imm:$src)]>;

def addri   : InstARM<(ops IntRegs:$dst, IntRegs:$a, i32imm:$b),
                       "add $dst, $a, $b",
		       [(set IntRegs:$dst, (add IntRegs:$a, imm:$b))]>;
