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
  let NumMIOperands = 3;
  let MIOperandInfo = (ops ptr_rc, ptr_rc, i32imm);
}

def memri : Operand<iPTR> {
  let PrintMethod = "printMemRegImm";
  let NumMIOperands = 2;
  let MIOperandInfo = (ops i32imm, ptr_rc);
}

// Define ARM specific addressing mode.
//Addressing Mode 1: data processing operands
def addr_mode1 : ComplexPattern<iPTR, 3, "SelectAddrMode1", [imm, sra, shl, srl]>;

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

def SDTarmfmstat   : SDTypeProfile<0, 0, []>;
def armfmstat      : SDNode<"ARMISD::FMSTAT", SDTarmfmstat, [SDNPInFlag, SDNPOutFlag]>;

def SDTarmbr       : SDTypeProfile<0, 2, [SDTCisVT<0, OtherVT>, SDTCisVT<1, i32>]>;
def armbr          : SDNode<"ARMISD::BR", SDTarmbr, [SDNPHasChain, SDNPInFlag]>;

def SDTVoidBinOp : SDTypeProfile<0, 2, [SDTCisSameAs<0, 1>]>;
def armcmp       : SDNode<"ARMISD::CMP",  SDTVoidBinOp, [SDNPOutFlag]>;

def armfsitos      : SDNode<"ARMISD::FSITOS", SDTUnaryOp>;
def armftosis      : SDNode<"ARMISD::FTOSIS", SDTUnaryOp>;
def armfsitod      : SDNode<"ARMISD::FSITOD", SDTUnaryOp>;
def armftosid      : SDNode<"ARMISD::FTOSID", SDTUnaryOp>;
def armfuitos      : SDNode<"ARMISD::FUITOS", SDTUnaryOp>;
def armftouis      : SDNode<"ARMISD::FTOUIS", SDTUnaryOp>;
def armfuitod      : SDNode<"ARMISD::FUITOD", SDTUnaryOp>;
def armftouid      : SDNode<"ARMISD::FTOUID", SDTUnaryOp>;

def SDTarmfmrrd    : SDTypeProfile<0, 3, [SDTCisInt<0>, SDTCisInt<1>, SDTCisFP<2>]>;
def armfmrrd       : SDNode<"ARMISD::FMRRD", SDTarmfmrrd,
                            [SDNPHasChain, SDNPOptInFlag, SDNPOutFlag]>;

def SDTarmfmdrr    : SDTypeProfile<1, 2, [SDTCisFP<0>, SDTCisInt<1>, SDTCisInt<2>]>;
def armfmdrr       : SDNode<"ARMISD::FMDRR", SDTarmfmdrr, []>;

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

def ADCS    : InstARM<(ops IntRegs:$dst, IntRegs:$a, op_addr_mode1:$b),
                       "adcs $dst, $a, $b",
		       [(set IntRegs:$dst, (adde IntRegs:$a, addr_mode1:$b))]>;

def ADDS    : InstARM<(ops IntRegs:$dst, IntRegs:$a, op_addr_mode1:$b),
                       "adds $dst, $a, $b",
		       [(set IntRegs:$dst, (addc IntRegs:$a, addr_mode1:$b))]>;

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

// Floating Point Compare
def fcmpes  : InstARM<(ops FPRegs:$a, FPRegs:$b),
	               "fcmpes $a, $b",
		       [(armcmp FPRegs:$a, FPRegs:$b)]>;

def fcmped  : InstARM<(ops DFPRegs:$a, DFPRegs:$b),
	               "fcmped $a, $b",
		       [(armcmp DFPRegs:$a, DFPRegs:$b)]>;

// Floating Point Conversion
// We use bitconvert for moving the data between the register classes.
// The format conversion is done with ARM specific nodes

def FMSR    : InstARM<(ops FPRegs:$dst, IntRegs:$src),
                       "fmsr $dst, $src", [(set FPRegs:$dst, (bitconvert IntRegs:$src))]>;

def FMRS    : InstARM<(ops IntRegs:$dst, FPRegs:$src),
                       "fmrs $dst, $src", [(set IntRegs:$dst, (bitconvert FPRegs:$src))]>;

def FMRRD   : InstARM<(ops IntRegs:$i0, IntRegs:$i1, DFPRegs:$src),
                       "fmrrd $i0, $i1, $src", [(armfmrrd IntRegs:$i0, IntRegs:$i1, DFPRegs:$src)]>;

def FMDRR   : InstARM<(ops DFPRegs:$dst, IntRegs:$i0, IntRegs:$i1),
                       "fmdrr $dst, $i0, $i1", [(set DFPRegs:$dst, (armfmdrr IntRegs:$i0, IntRegs:$i1))]>;

def FSITOS  : InstARM<(ops FPRegs:$dst, FPRegs:$src),
                       "fsitos $dst, $src", [(set FPRegs:$dst, (armfsitos FPRegs:$src))]>;

def FTOSIS  : InstARM<(ops FPRegs:$dst, FPRegs:$src),
                       "ftosis $dst, $src", [(set FPRegs:$dst, (armftosis FPRegs:$src))]>;

def FSITOD  : InstARM<(ops DFPRegs:$dst, FPRegs:$src),
                       "fsitod $dst, $src", [(set DFPRegs:$dst, (armfsitod FPRegs:$src))]>;

def FTOSID  : InstARM<(ops FPRegs:$dst, DFPRegs:$src),
                       "ftosid $dst, $src", [(set FPRegs:$dst, (armftosid DFPRegs:$src))]>;

def FUITOS  : InstARM<(ops FPRegs:$dst, FPRegs:$src),
                       "fuitos $dst, $src", [(set FPRegs:$dst, (armfuitos FPRegs:$src))]>;

def FTOUIS  : InstARM<(ops FPRegs:$dst, FPRegs:$src),
                       "ftouis $dst, $src", [(set FPRegs:$dst, (armftouis FPRegs:$src))]>;

def FUITOD  : InstARM<(ops DFPRegs:$dst, FPRegs:$src),
                       "fuitod $dst, $src", [(set DFPRegs:$dst, (armfuitod FPRegs:$src))]>;

def FTOUID  : InstARM<(ops FPRegs:$dst, DFPRegs:$src),
                       "ftouid $dst, $src", [(set FPRegs:$dst, (armftouid DFPRegs:$src))]>;

def FCVTDS  : InstARM<(ops DFPRegs:$dst, FPRegs:$src),
                       "fcvtds $dst, $src", [(set DFPRegs:$dst, (fextend FPRegs:$src))]>;

def FCVTSD  : InstARM<(ops FPRegs:$dst, DFPRegs:$src),
                       "fcvtsd $dst, $src", [(set FPRegs:$dst, (fround DFPRegs:$src))]>;

def FMSTAT  : InstARM<(ops ), "fmstat", [(armfmstat)]>;

// Floating Point Arithmetic
def FADDS   : InstARM<(ops FPRegs:$dst, FPRegs:$a, FPRegs:$b),
                       "fadds $dst, $a, $b",
		       [(set FPRegs:$dst, (fadd FPRegs:$a, FPRegs:$b))]>;

def FADDD   : InstARM<(ops DFPRegs:$dst, DFPRegs:$a, DFPRegs:$b),
                       "faddd $dst, $a, $b",
		       [(set DFPRegs:$dst, (fadd DFPRegs:$a, DFPRegs:$b))]>;

def FSUBS   : InstARM<(ops FPRegs:$dst, FPRegs:$a, FPRegs:$b),
                       "fsubs $dst, $a, $b",
		       [(set FPRegs:$dst, (fsub FPRegs:$a, FPRegs:$b))]>;

def FSUBD   : InstARM<(ops DFPRegs:$dst, DFPRegs:$a, DFPRegs:$b),
                       "fsubd $dst, $a, $b",
		       [(set DFPRegs:$dst, (fsub DFPRegs:$a, DFPRegs:$b))]>;

def FMULS   : InstARM<(ops FPRegs:$dst, FPRegs:$a, FPRegs:$b),
                       "fmuls $dst, $a, $b",
		       [(set FPRegs:$dst, (fmul FPRegs:$a, FPRegs:$b))]>;

def FMULD   : InstARM<(ops DFPRegs:$dst, DFPRegs:$a, DFPRegs:$b),
                       "fmuld $dst, $a, $b",
		       [(set DFPRegs:$dst, (fmul DFPRegs:$a, DFPRegs:$b))]>;


// Floating Point Load
def FLDS  : InstARM<(ops FPRegs:$dst, IntRegs:$addr),
                     "flds $dst, $addr",
                     [(set FPRegs:$dst, (load IntRegs:$addr))]>;

def FLDD  : InstARM<(ops DFPRegs:$dst, IntRegs:$addr),
                     "fldd $dst, $addr",
                     [(set DFPRegs:$dst, (load IntRegs:$addr))]>;
