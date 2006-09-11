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
def op_addr_mode1 : Operand<iPTR> {
  let PrintMethod = "printAddrMode1";
  let NumMIOperands = 1;
  let MIOperandInfo = (ops ptr_rc);
}

def memri : Operand<iPTR> {
  let PrintMethod = "printMemRegImm";
  let NumMIOperands = 2;
  let MIOperandInfo = (ops i32imm, ptr_rc);
}

// Define ARM specific addressing mode.
//Addressing Mode 1: data processing operands
def addr_mode1 : ComplexPattern<iPTR, 1, "SelectAddrMode1", [imm]>;

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

def brtarget : Operand<OtherVT>;

// Operand for printing out a condition code.
let PrintMethod = "printCCOperand" in
  def CCOp : Operand<i32>;

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

def SDTarmselect   : SDTypeProfile<1, 3, [SDTCisSameAs<0, 1>, SDTCisSameAs<0, 2>, SDTCisInt<0>, SDTCisVT<2, i32>]>;

def armselect      : SDNode<"ARMISD::SELECT", SDTarmselect, [SDNPInFlag, SDNPOutFlag]>;

def SDTarmbr       : SDTypeProfile<0, 2, [SDTCisVT<0, OtherVT>, SDTCisVT<1, i32>]>;
def armbr          : SDNode<"ARMISD::BR", SDTarmbr, [SDNPHasChain, SDNPInFlag]>;

def SDTVoidBinOp : SDTypeProfile<0, 2, [SDTCisSameAs<0, 1>]>;
def armcmp       : SDNode<"ARMISD::CMP",  SDTVoidBinOp, [SDNPOutFlag]>;

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

def MOV   : InstARM<(ops IntRegs:$dst, op_addr_mode1:$src),
                    "mov $dst, $src", [(set IntRegs:$dst, addr_mode1:$src)]>;

def ADD     : InstARM<(ops IntRegs:$dst, IntRegs:$a, op_addr_mode1:$b),
                       "add $dst, $a, $b",
		       [(set IntRegs:$dst, (add IntRegs:$a, addr_mode1:$b))]>;

// "LEA" forms of add
def lea_addri : InstARM<(ops IntRegs:$dst, memri:$addr),
			 "add $dst, ${addr:arith}",
                	 [(set IntRegs:$dst, iaddr:$addr)]>;


def SUB     : InstARM<(ops IntRegs:$dst, IntRegs:$a, op_addr_mode1:$b),
                       "sub $dst, $a, $b",
		       [(set IntRegs:$dst, (sub IntRegs:$a, addr_mode1:$b))]>;

def AND     : InstARM<(ops IntRegs:$dst, IntRegs:$a, op_addr_mode1:$b),
                       "and $dst, $a, $b",
		       [(set IntRegs:$dst, (and IntRegs:$a, addr_mode1:$b))]>;

// All arm data processing instructions have a shift. Maybe we don't have
// to implement this
def SHL : InstARM<(ops IntRegs:$dst, IntRegs:$a, IntRegs:$b),
		       "mov $dst, $a, lsl $b",
		       [(set IntRegs:$dst, (shl IntRegs:$a, IntRegs:$b))]>;

def SRA : InstARM<(ops IntRegs:$dst, IntRegs:$a, IntRegs:$b),
		       "mov $dst, $a, asr $b",
		       [(set IntRegs:$dst, (sra IntRegs:$a, IntRegs:$b))]>;

def SRL : InstARM<(ops IntRegs:$dst, IntRegs:$a, IntRegs:$b),
		       "mov $dst, $a, lsr $b",
		       [(set IntRegs:$dst, (srl IntRegs:$a, IntRegs:$b))]>;


def EOR     : InstARM<(ops IntRegs:$dst, IntRegs:$a, op_addr_mode1:$b),
                       "eor $dst, $a, $b",
		       [(set IntRegs:$dst, (xor IntRegs:$a, addr_mode1:$b))]>;

def ORR     : InstARM<(ops IntRegs:$dst, IntRegs:$a, op_addr_mode1:$b),
                       "orr $dst, $a, $b",
		       [(set IntRegs:$dst, (or IntRegs:$a, addr_mode1:$b))]>;

let isTwoAddress = 1 in {
  def movcond : InstARM<(ops IntRegs:$dst, IntRegs:$false,
			 op_addr_mode1:$true, CCOp:$cc),
	                 "mov$cc $dst, $true",
		         [(set IntRegs:$dst, (armselect addr_mode1:$true,
			   IntRegs:$false, imm:$cc))]>;
}

def MUL     : InstARM<(ops IntRegs:$dst, IntRegs:$a, IntRegs:$b),
                       "mul $dst, $a, $b",
		       [(set IntRegs:$dst, (mul IntRegs:$a, IntRegs:$b))]>;

def bcond      : InstARM<(ops brtarget:$dst, CCOp:$cc),
		         "b$cc $dst",
		         [(armbr bb:$dst, imm:$cc)]>;

def b      : InstARM<(ops brtarget:$dst),
		         "b $dst",
		         [(br bb:$dst)]>;

def cmp      : InstARM<(ops IntRegs:$a, op_addr_mode1:$b),
	               "cmp $a, $b",
		       [(armcmp IntRegs:$a, addr_mode1:$b)]>;
