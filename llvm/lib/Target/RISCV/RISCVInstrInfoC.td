//===- RISCVInstrInfoC.td - Compressed RISCV instructions -*- tblgen-*-----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

include "RISCVInstrFormatsC.td"

//===----------------------------------------------------------------------===//
// Operand definitions.
//===----------------------------------------------------------------------===//

// A 7-bit unsigned immediate where the least significant two bits are zero.
def uimm7_lsb00 : Operand<XLenVT>,
                  ImmLeaf<XLenVT, [{return isShiftedUInt<5, 2>(Imm);}]> {
  let ParserMatchClass = UImmAsmOperand<7, "Lsb00">;
  let EncoderMethod = "getImmOpValue";
  let DecoderMethod = "decodeUImmOperand<7>";
}

// A 8-bit unsigned immediate where the least significant two bits are zero.
def uimm8_lsb00 : Operand<XLenVT>,
                  ImmLeaf<XLenVT, [{return isShiftedUInt<6, 2>(Imm);}]> {
  let ParserMatchClass = UImmAsmOperand<8, "Lsb00">;
  let EncoderMethod = "getImmOpValue";
  let DecoderMethod = "decodeUImmOperand<8>";
}

// A 8-bit unsigned immediate where the least significant three bits are zero.
def uimm8_lsb000 : Operand<XLenVT>,
                   ImmLeaf<XLenVT, [{return isShiftedUInt<5, 3>(Imm);}]> {
  let ParserMatchClass = UImmAsmOperand<8, "Lsb000">;
  let EncoderMethod = "getImmOpValue";
  let DecoderMethod = "decodeUImmOperand<8>";
}

// A 9-bit unsigned immediate where the least significant three bits are zero.
def uimm9_lsb000 : Operand<XLenVT>,
                   ImmLeaf<XLenVT, [{return isShiftedUInt<6, 3>(Imm);}]> {
  let ParserMatchClass = UImmAsmOperand<9, "Lsb000">;
  let EncoderMethod = "getImmOpValue";
  let DecoderMethod = "decodeUImmOperand<9>";
}

//===----------------------------------------------------------------------===//
// Instruction Class Templates
//===----------------------------------------------------------------------===//

let hasSideEffects = 0, mayLoad = 1, mayStore = 0 in
class CStackLoad<bits<3> funct3, string OpcodeStr,
                 RegisterClass cls, DAGOperand opnd> :
      RVInst16CI<funct3, 0b10, (outs cls:$rd), (ins SP:$rs1, opnd:$imm),
                 OpcodeStr, "$rd, ${imm}(${rs1})">;

let hasSideEffects = 0, mayLoad = 0, mayStore = 1 in
class CStackStore<bits<3> funct3, string OpcodeStr,
                  RegisterClass cls, DAGOperand opnd> :
      RVInst16CSS<funct3, 0b10, (outs), (ins cls:$rs2, SP:$rs1, opnd:$imm),
                  OpcodeStr, "$rs2, ${imm}(${rs1})">;

let hasSideEffects = 0, mayLoad = 1, mayStore = 0 in
class CLoad_ri<bits<3> funct3, string OpcodeStr,
               RegisterClass cls, DAGOperand opnd> :
      RVInst16CL<funct3, 0b00, (outs cls:$rd), (ins cls:$rs1, opnd:$imm),
                 OpcodeStr, "$rd, ${imm}(${rs1})">;

let hasSideEffects = 0, mayLoad = 0, mayStore = 1 in
class CStore_rri<bits<3> funct3, string OpcodeStr,
                 RegisterClass cls, DAGOperand opnd> :
      RVInst16CS<funct3, 0b00, (outs), (ins cls:$rs2, cls:$rs1, opnd:$imm),
                 OpcodeStr, "$rs2, ${imm}(${rs1})">;

//===----------------------------------------------------------------------===//
// Instructions
//===----------------------------------------------------------------------===//

let Predicates = [HasStdExtC] in {

def CLW : CLoad_ri<0b010, "c.lw", GPRC, uimm7_lsb00> {
  bits<7> imm;
  let Inst{12-10} = imm{5-3};
  let Inst{6} = imm{2};
  let Inst{5} = imm{6};
}

def CLD : CLoad_ri<0b011, "c.ld", GPRC, uimm8_lsb000>,
          Requires<[IsRV64]> {
  bits<8> imm;
  let Inst{12-10} = imm{5-3};
  let Inst{6-5} = imm{7-6};
}

def CSW : CStore_rri<0b110, "c.sw", GPRC, uimm7_lsb00> {
  bits<7> imm;
  let Inst{12-10} = imm{5-3};
  let Inst{6} = imm{2};
  let Inst{5} = imm{6};
}

def CSD : CStore_rri<0b111, "c.sd", GPRC, uimm8_lsb000>,
          Requires<[IsRV64]> {
  bits<8> imm;
  let Inst{12-10} = imm{5-3};
  let Inst{6-5} = imm{7-6};
}

def CLWSP : CStackLoad<0b010, "c.lwsp", GPRNoX0, uimm8_lsb00> {
  let Inst{6-4} = imm{4-2};
  let Inst{3-2} = imm{7-6};
}

def CLDSP : CStackLoad<0b011, "c.ldsp", GPRNoX0, uimm9_lsb000>,
            Requires<[IsRV64]> {
  let Inst{6-5} = imm{4-3};
  let Inst{4-2} = imm{8-6};
}

def CSWSP : CStackStore<0b110, "c.swsp", GPR, uimm8_lsb00> {
  let Inst{12-9} = imm{5-2};
  let Inst{8-7}  = imm{7-6};
}

def CSDSP : CStackStore<0b111, "c.sdsp", GPR, uimm9_lsb000>,
            Requires<[IsRV64]> {
  let Inst{12-10} = imm{5-3};
  let Inst{9-7}   = imm{8-6};
}

} // Predicates = [HasStdExtC]
