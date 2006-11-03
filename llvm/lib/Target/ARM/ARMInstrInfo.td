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
  let MIOperandInfo = (ops ptr_rc, ptr_rc, i32imm);
}

def op_addr_mode5 : Operand<iPTR> {
  let PrintMethod = "printAddrMode5";
  let MIOperandInfo = (ops ptr_rc, i32imm);
}

def memri : Operand<iPTR> {
  let PrintMethod = "printMemRegImm";
  let MIOperandInfo = (ops i32imm, ptr_rc);
}

// Define ARM specific addressing mode.
//Addressing Mode 1: data processing operands
def addr_mode1 : ComplexPattern<iPTR, 3, "SelectAddrMode1", [imm, sra, shl, srl],
                                []>;

//Addressing Mode 5: VFP load/store
def addr_mode5 : ComplexPattern<iPTR, 2, "SelectAddrMode5", [], []>;

//register plus/minus 12 bit offset
def iaddr  : ComplexPattern<iPTR, 2, "SelectAddrRegImm", [frameindex], []>;
//register plus scaled register
//def raddr  : ComplexPattern<iPTR, 2, "SelectAddrRegReg", [], []>;

//===----------------------------------------------------------------------===//
// Instruction Class Templates
//===----------------------------------------------------------------------===//
class InstARM<dag ops, string asmstr, list<dag> pattern> : Instruction {
  let Namespace = "ARM";

  dag OperandList = ops;
  let AsmString   = asmstr;
  let Pattern = pattern;
}

class IntBinOp<string OpcStr, SDNode OpNode> :
        InstARM<(ops IntRegs:$dst, IntRegs:$a, IntRegs:$b),
                 !strconcat(OpcStr, " $dst, $a, $b"),
                 [(set IntRegs:$dst, (OpNode IntRegs:$a, IntRegs:$b))]>;

class FPBinOp<string OpcStr, SDNode OpNode> :
        InstARM<(ops FPRegs:$dst, FPRegs:$a, FPRegs:$b),
                 !strconcat(OpcStr, " $dst, $a, $b"),
                 [(set FPRegs:$dst, (OpNode FPRegs:$a, FPRegs:$b))]>;

class DFPBinOp<string OpcStr, SDNode OpNode> :
        InstARM<(ops DFPRegs:$dst, DFPRegs:$a, DFPRegs:$b),
                 !strconcat(OpcStr, " $dst, $a, $b"),
                 [(set DFPRegs:$dst, (OpNode DFPRegs:$a, DFPRegs:$b))]>;

class FPUnaryOp<string OpcStr, SDNode OpNode> :
        InstARM<(ops FPRegs:$dst, FPRegs:$src),
                 !strconcat(OpcStr, " $dst, $src"),
                 [(set FPRegs:$dst, (OpNode FPRegs:$src))]>;

class DFPUnaryOp<string OpcStr, SDNode OpNode> :
        InstARM<(ops DFPRegs:$dst, DFPRegs:$src),
                 !strconcat(OpcStr, " $dst, $src"),
                 [(set DFPRegs:$dst, (OpNode DFPRegs:$src))]>;

class Addr1BinOp<string OpcStr, SDNode OpNode> :
        InstARM<(ops IntRegs:$dst, IntRegs:$a, op_addr_mode1:$b),
                 !strconcat(OpcStr, " $dst, $a, $b"),
                 [(set IntRegs:$dst, (OpNode IntRegs:$a, addr_mode1:$b))]>;

//===----------------------------------------------------------------------===//
// Instructions
//===----------------------------------------------------------------------===//

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
                            [(callseq_end imm:$amt)]>, Imp<[R13],[R13]>;

def ADJCALLSTACKDOWN : InstARM<(ops i32imm:$amt),
                               "!ADJCALLSTACKDOWN $amt",
                               [(callseq_start imm:$amt)]>, Imp<[R13],[R13]>;

def IMPLICIT_DEF_Int : InstARM<(ops IntRegs:$dst),
                               "@IMPLICIT_DEF $dst",
                               [(set IntRegs:$dst, (undef))]>;
def IMPLICIT_DEF_FP  : InstARM<(ops FPRegs:$dst), "@IMPLICIT_DEF $dst",
                               [(set FPRegs:$dst, (undef))]>;
def IMPLICIT_DEF_DFP : InstARM<(ops DFPRegs:$dst), "@IMPLICIT_DEF $dst",
                               [(set DFPRegs:$dst, (undef))]>;

let isReturn = 1 in {
  def bx: InstARM<(ops), "bx r14", [(retflag)]>;
}

let noResults = 1, Defs = [R0, R1, R2, R3, R14] in {
  def bl: InstARM<(ops i32imm:$func, variable_ops), "bl $func", []>;
  def blx     : InstARM<(ops IntRegs:$func, variable_ops), "blx $func", [(ARMcall IntRegs:$func)]>;
}

def ldr   : InstARM<(ops IntRegs:$dst, memri:$addr),
                     "ldr $dst, $addr",
                     [(set IntRegs:$dst, (load iaddr:$addr))]>;

def LDRB    : InstARM<(ops IntRegs:$dst, IntRegs:$addr),
                       "ldrb $dst, [$addr]",
                       [(set IntRegs:$dst, (zextloadi8 IntRegs:$addr))]>;

def LDRSB   : InstARM<(ops IntRegs:$dst, IntRegs:$addr),
                       "ldrsb $dst, [$addr]",
                       [(set IntRegs:$dst, (sextloadi8 IntRegs:$addr))]>;

def LDRH    : InstARM<(ops IntRegs:$dst, IntRegs:$addr),
                       "ldrh $dst, [$addr]",
                       [(set IntRegs:$dst, (zextloadi16 IntRegs:$addr))]>;

def LDRSH   : InstARM<(ops IntRegs:$dst, IntRegs:$addr),
                       "ldrsh $dst, [$addr]",
                       [(set IntRegs:$dst, (sextloadi16 IntRegs:$addr))]>;

def str  : InstARM<(ops IntRegs:$src, memri:$addr),
                    "str $src, $addr",
                    [(store IntRegs:$src, iaddr:$addr)]>;

def STRB    : InstARM<(ops IntRegs:$src, IntRegs:$addr),
                       "strb $src, [$addr]",
                       [(truncstorei8 IntRegs:$src, IntRegs:$addr)]>;

def STRH    : InstARM<(ops IntRegs:$src, IntRegs:$addr),
                       "strh $src, [$addr]",
                       [(truncstorei16 IntRegs:$src, IntRegs:$addr)]>;

def MOV   : InstARM<(ops IntRegs:$dst, op_addr_mode1:$src),
                    "mov $dst, $src", [(set IntRegs:$dst, addr_mode1:$src)]>;

def ADD     : Addr1BinOp<"add",  add>;
def ADCS    : Addr1BinOp<"adcs", adde>;
def ADDS    : Addr1BinOp<"adds", addc>;

// "LEA" forms of add
def lea_addri : InstARM<(ops IntRegs:$dst, memri:$addr),
			 "add $dst, ${addr:arith}",
                	 [(set IntRegs:$dst, iaddr:$addr)]>;


def SUB     : Addr1BinOp<"sub",  sub>;
def SBCS    : Addr1BinOp<"sbcs", sube>;
def SUBS    : Addr1BinOp<"subs", subc>;
def AND     : Addr1BinOp<"and",  and>;
def EOR     : Addr1BinOp<"eor",  xor>;
def ORR     : Addr1BinOp<"orr",  or>;

let isTwoAddress = 1 in {
  def movcond : InstARM<(ops IntRegs:$dst, IntRegs:$false,
			 op_addr_mode1:$true, CCOp:$cc),
	                 "mov$cc $dst, $true",
		         [(set IntRegs:$dst, (armselect addr_mode1:$true,
			   IntRegs:$false, imm:$cc))]>;
}

def MUL     : IntBinOp<"mul", mul>;

let Defs = [R0] in {
  def SMULL   : IntBinOp<"smull r12,", mulhs>;
  def UMULL   : IntBinOp<"umull r12,", mulhu>;
}

let isTerminator = 1, isBranch = 1 in {
  def bcond   : InstARM<(ops brtarget:$dst, CCOp:$cc),
	                    "b$cc $dst",
		            [(armbr bb:$dst, imm:$cc)]>;

  def b       : InstARM<(ops brtarget:$dst),
                        "b $dst",
                        [(br bb:$dst)]>;
}

def cmp      : InstARM<(ops IntRegs:$a, op_addr_mode1:$b),
	               "cmp $a, $b",
		       [(armcmp IntRegs:$a, addr_mode1:$b)]>;

// Floating Point Compare
def fcmps   : InstARM<(ops FPRegs:$a, FPRegs:$b),
	               "fcmps $a, $b",
		       [(armcmp FPRegs:$a, FPRegs:$b)]>;

def fcmpd   : InstARM<(ops DFPRegs:$a, DFPRegs:$b),
	               "fcmpd $a, $b",
		       [(armcmp DFPRegs:$a, DFPRegs:$b)]>;

// Floating Point Copy
def FCPYS   : InstARM<(ops FPRegs:$dst, FPRegs:$src), "fcpys $dst, $src", []>;

def FCPYD   : InstARM<(ops DFPRegs:$dst, DFPRegs:$src), "fcpyd $dst, $src", []>;

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
def FADDS   : FPBinOp<"fadds",  fadd>;
def FADDD   : DFPBinOp<"faddd", fadd>;
def FSUBS   : FPBinOp<"fsubs",  fsub>;
def FSUBD   : DFPBinOp<"fsubd", fsub>;

def FNEGS   : FPUnaryOp<"fnegs",  fneg>;
def FNEGD   : DFPUnaryOp<"fnegd", fneg>;
def FABSS   : FPUnaryOp<"fabss",  fabs>;
def FABSD   : DFPUnaryOp<"fabsd", fabs>;

def FMULS   : FPBinOp<"fmuls", fmul>;
def FMULD   : DFPBinOp<"fmuld", fmul>;
def FDIVS   : FPBinOp<"fdivs", fdiv>;
def FDIVD   : DFPBinOp<"fdivd", fdiv>;

// Floating Point Load
def FLDS  : InstARM<(ops FPRegs:$dst, op_addr_mode5:$addr),
                     "flds $dst, $addr",
                     [(set FPRegs:$dst, (load addr_mode5:$addr))]>;

def FLDD  : InstARM<(ops DFPRegs:$dst, op_addr_mode5:$addr),
                     "fldd $dst, $addr",
                     [(set DFPRegs:$dst, (load addr_mode5:$addr))]>;

// Floating Point Store
def FSTS    : InstARM<(ops FPRegs:$src, op_addr_mode5:$addr),
                       "fsts $src, $addr",
                       [(store FPRegs:$src, addr_mode5:$addr)]>;

def FSTD    : InstARM<(ops DFPRegs:$src, op_addr_mode5:$addr),
                       "fstd $src, $addr",
                       [(store DFPRegs:$src, addr_mode5:$addr)]>;

def : Pat<(ARMcall tglobaladdr:$dst),
          (bl tglobaladdr:$dst)>;

def : Pat<(ARMcall texternalsym:$dst),
          (bl texternalsym:$dst)>;

def : Pat<(extloadi8 IntRegs:$addr),
          (LDRB IntRegs:$addr)>;
def : Pat<(extloadi16 IntRegs:$addr),
          (LDRH IntRegs:$addr)>;

// zextload bool -> zextload byte
def : Pat<(i32 (zextloadi1 IntRegs:$addr)), (LDRB IntRegs:$addr)>;
def : Pat<(i32 (zextloadi1 IntRegs:$addr)), (LDRB IntRegs:$addr)>;

// truncstore bool -> truncstore byte.
def : Pat<(truncstorei1 IntRegs:$src, IntRegs:$addr),
          (STRB IntRegs:$addr, IntRegs:$src)>;
def : Pat<(truncstorei1 IntRegs:$src, IntRegs:$addr),
          (STRB IntRegs:$addr, IntRegs:$src)>;
