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

// Address operands
def memri : Operand<iPTR> {
  let PrintMethod = "printMemRegImm";
  let NumMIOperands = 2;
  let MIOperandInfo = (ops i32imm, ptr_rc);
}

// Define ARM specific addressing mode.
//register plus/minus 12 bit offset
def iaddr  : ComplexPattern<iPTR, 2, "SelectAddrRegImm", [frameindex]>;
//register plus scaled register
//def raddr  : ComplexPattern<iPTR, 2, "SelectAddrRegReg", []>;

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
def callseq_start  : SDNode<"ISD::CALLSEQ_START", SDT_ARMCallSeq,
    		             [SDNPHasChain, SDNPOutFlag]>;
def callseq_end    : SDNode<"ISD::CALLSEQ_END",   SDT_ARMCallSeq,
    		             [SDNPHasChain, SDNPOutFlag]>;

def SDT_ARMcall    : SDTypeProfile<0, -1, [SDTCisInt<0>]>;
def ARMcall        : SDNode<"ARMISD::CALL", SDT_ARMcall,
                           [SDNPHasChain, SDNPOptInFlag, SDNPOutFlag]>;
def retflag        : SDNode<"ARMISD::RET_FLAG", SDTRet,
	                   [SDNPHasChain, SDNPOptInFlag]>;

def ADJCALLSTACKUP : InstARM<(ops i32imm:$amt),
                            "!ADJCALLSTACKUP $amt",
                            [(callseq_end imm:$amt)]>;

def ADJCALLSTACKDOWN : InstARM<(ops i32imm:$amt),
                               "!ADJCALLSTACKDOWN $amt",
                               [(callseq_start imm:$amt)]>;

let isReturn = 1 in {
  def bx: InstARM<(ops), "bx r14", [(retflag)]>;
}

let  Defs = [R0, R1, R2, R3, R14] in {
  def bl: InstARM<(ops i32imm:$func, variable_ops), "bl $func", [(ARMcall tglobaladdr:$func)]>;
}

def ldr   : InstARM<(ops IntRegs:$dst, memri:$addr),
                     "ldr $dst, $addr",
                     [(set IntRegs:$dst, (load iaddr:$addr))]>;

def str  : InstARM<(ops IntRegs:$src, memri:$addr),
                    "str $src, $addr",
                    [(store IntRegs:$src, iaddr:$addr)]>;

def movrr   : InstARM<(ops IntRegs:$dst, IntRegs:$src),
                       "mov $dst, $src", []>;

def movri   : InstARM<(ops IntRegs:$dst, i32imm:$src),
                       "mov $dst, $src", [(set IntRegs:$dst, imm:$src)]>;

def addri   : InstARM<(ops IntRegs:$dst, IntRegs:$a, i32imm:$b),
                       "add $dst, $a, $b",
		       [(set IntRegs:$dst, (add IntRegs:$a, imm:$b))]>;

// "LEA" forms of add
def lea_addri : InstARM<(ops IntRegs:$dst, memri:$addr),
			 "add $dst, ${addr:arith}",
                	 [(set IntRegs:$dst, iaddr:$addr)]>;


def subri   : InstARM<(ops IntRegs:$dst, IntRegs:$a, i32imm:$b),
                       "sub $dst, $a, $b",
		       [(set IntRegs:$dst, (sub IntRegs:$a, imm:$b))]>;
