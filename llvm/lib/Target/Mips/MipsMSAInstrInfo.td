//===- MipsMSAInstrInfo.td - MSA ASE instructions -*- tablegen ------------*-=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file describes Mips MSA ASE instructions.
//
//===----------------------------------------------------------------------===//

def immSExt5 : ImmLeaf<i32, [{return isInt<5>(Imm);}]>;
def immSExt10: ImmLeaf<i32, [{return isInt<10>(Imm);}]>;

def uimm3 : Operand<i32> {
  let PrintMethod = "printUnsignedImm";
}

def uimm4 : Operand<i32> {
  let PrintMethod = "printUnsignedImm";
}

def uimm6 : Operand<i32> {
  let PrintMethod = "printUnsignedImm";
}

def uimm8 : Operand<i32> {
  let PrintMethod = "printUnsignedImm";
}

def simm5 : Operand<i32>;

def simm10 : Operand<i32>;

// Instruction encoding.
class ADD_A_B_ENC : MSA_3R_FMT<0b000, 0b00, 0b010000>;
class ADD_A_H_ENC : MSA_3R_FMT<0b000, 0b01, 0b010000>;
class ADD_A_W_ENC : MSA_3R_FMT<0b000, 0b10, 0b010000>;
class ADD_A_D_ENC : MSA_3R_FMT<0b000, 0b11, 0b010000>;

class ADDS_A_B_ENC : MSA_3R_FMT<0b001, 0b00, 0b010000>;
class ADDS_A_H_ENC : MSA_3R_FMT<0b001, 0b01, 0b010000>;
class ADDS_A_W_ENC : MSA_3R_FMT<0b001, 0b10, 0b010000>;
class ADDS_A_D_ENC : MSA_3R_FMT<0b001, 0b11, 0b010000>;

class ADDS_S_B_ENC : MSA_3R_FMT<0b010, 0b00, 0b010000>;
class ADDS_S_H_ENC : MSA_3R_FMT<0b010, 0b01, 0b010000>;
class ADDS_S_W_ENC : MSA_3R_FMT<0b010, 0b10, 0b010000>;
class ADDS_S_D_ENC : MSA_3R_FMT<0b010, 0b11, 0b010000>;

class ADDS_U_B_ENC : MSA_3R_FMT<0b011, 0b00, 0b010000>;
class ADDS_U_H_ENC : MSA_3R_FMT<0b011, 0b01, 0b010000>;
class ADDS_U_W_ENC : MSA_3R_FMT<0b011, 0b10, 0b010000>;
class ADDS_U_D_ENC : MSA_3R_FMT<0b011, 0b11, 0b010000>;

class ADDV_B_ENC : MSA_3R_FMT<0b000, 0b00, 0b001110>;
class ADDV_H_ENC : MSA_3R_FMT<0b000, 0b01, 0b001110>;
class ADDV_W_ENC : MSA_3R_FMT<0b000, 0b10, 0b001110>;
class ADDV_D_ENC : MSA_3R_FMT<0b000, 0b11, 0b001110>;

class ADDVI_B_ENC : MSA_I5_FMT<0b000, 0b00, 0b000110>;
class ADDVI_H_ENC : MSA_I5_FMT<0b000, 0b01, 0b000110>;
class ADDVI_W_ENC : MSA_I5_FMT<0b000, 0b10, 0b000110>;
class ADDVI_D_ENC : MSA_I5_FMT<0b000, 0b11, 0b000110>;

class ANDI_B_ENC : MSA_I8_FMT<0b00, 0b000000>;

class ASUB_S_B_ENC : MSA_3R_FMT<0b100, 0b00, 0b010001>;
class ASUB_S_H_ENC : MSA_3R_FMT<0b100, 0b01, 0b010001>;
class ASUB_S_W_ENC : MSA_3R_FMT<0b100, 0b10, 0b010001>;
class ASUB_S_D_ENC : MSA_3R_FMT<0b100, 0b11, 0b010001>;

class ASUB_U_B_ENC : MSA_3R_FMT<0b101, 0b00, 0b010001>;
class ASUB_U_H_ENC : MSA_3R_FMT<0b101, 0b01, 0b010001>;
class ASUB_U_W_ENC : MSA_3R_FMT<0b101, 0b10, 0b010001>;
class ASUB_U_D_ENC : MSA_3R_FMT<0b101, 0b11, 0b010001>;

class AVE_S_B_ENC : MSA_3R_FMT<0b100, 0b00, 0b010000>;
class AVE_S_H_ENC : MSA_3R_FMT<0b100, 0b01, 0b010000>;
class AVE_S_W_ENC : MSA_3R_FMT<0b100, 0b10, 0b010000>;
class AVE_S_D_ENC : MSA_3R_FMT<0b100, 0b11, 0b010000>;

class AVE_U_B_ENC : MSA_3R_FMT<0b101, 0b00, 0b010000>;
class AVE_U_H_ENC : MSA_3R_FMT<0b101, 0b01, 0b010000>;
class AVE_U_W_ENC : MSA_3R_FMT<0b101, 0b10, 0b010000>;
class AVE_U_D_ENC : MSA_3R_FMT<0b101, 0b11, 0b010000>;

class AVER_S_B_ENC : MSA_3R_FMT<0b110, 0b00, 0b010000>;
class AVER_S_H_ENC : MSA_3R_FMT<0b110, 0b01, 0b010000>;
class AVER_S_W_ENC : MSA_3R_FMT<0b110, 0b10, 0b010000>;
class AVER_S_D_ENC : MSA_3R_FMT<0b110, 0b11, 0b010000>;

class AVER_U_B_ENC : MSA_3R_FMT<0b111, 0b00, 0b010000>;
class AVER_U_H_ENC : MSA_3R_FMT<0b111, 0b01, 0b010000>;
class AVER_U_W_ENC : MSA_3R_FMT<0b111, 0b10, 0b010000>;
class AVER_U_D_ENC : MSA_3R_FMT<0b111, 0b11, 0b010000>;

class BCLR_B_ENC : MSA_3R_FMT<0b011, 0b00, 0b001101>;
class BCLR_H_ENC : MSA_3R_FMT<0b011, 0b01, 0b001101>;
class BCLR_W_ENC : MSA_3R_FMT<0b011, 0b10, 0b001101>;
class BCLR_D_ENC : MSA_3R_FMT<0b011, 0b11, 0b001101>;

class BCLRI_B_ENC : MSA_BIT_B_FMT<0b011, 0b001001>;
class BCLRI_H_ENC : MSA_BIT_H_FMT<0b011, 0b001001>;
class BCLRI_W_ENC : MSA_BIT_W_FMT<0b011, 0b001001>;
class BCLRI_D_ENC : MSA_BIT_D_FMT<0b011, 0b001001>;

class BINSL_B_ENC : MSA_3R_FMT<0b110, 0b00, 0b001101>;
class BINSL_H_ENC : MSA_3R_FMT<0b110, 0b01, 0b001101>;
class BINSL_W_ENC : MSA_3R_FMT<0b110, 0b10, 0b001101>;
class BINSL_D_ENC : MSA_3R_FMT<0b110, 0b11, 0b001101>;

class BINSLI_B_ENC : MSA_BIT_B_FMT<0b110, 0b001001>;
class BINSLI_H_ENC : MSA_BIT_H_FMT<0b110, 0b001001>;
class BINSLI_W_ENC : MSA_BIT_W_FMT<0b110, 0b001001>;
class BINSLI_D_ENC : MSA_BIT_D_FMT<0b110, 0b001001>;

class BINSR_B_ENC : MSA_3R_FMT<0b111, 0b00, 0b001101>;
class BINSR_H_ENC : MSA_3R_FMT<0b111, 0b01, 0b001101>;
class BINSR_W_ENC : MSA_3R_FMT<0b111, 0b10, 0b001101>;
class BINSR_D_ENC : MSA_3R_FMT<0b111, 0b11, 0b001101>;

class BINSRI_B_ENC : MSA_BIT_B_FMT<0b111, 0b001001>;
class BINSRI_H_ENC : MSA_BIT_H_FMT<0b111, 0b001001>;
class BINSRI_W_ENC : MSA_BIT_W_FMT<0b111, 0b001001>;
class BINSRI_D_ENC : MSA_BIT_D_FMT<0b111, 0b001001>;

class BMNZI_B_ENC : MSA_I8_FMT<0b00, 0b000001>;

class BMZI_B_ENC : MSA_I8_FMT<0b01, 0b000001>;

class BNEG_B_ENC : MSA_3R_FMT<0b101, 0b00, 0b001101>;
class BNEG_H_ENC : MSA_3R_FMT<0b101, 0b01, 0b001101>;
class BNEG_W_ENC : MSA_3R_FMT<0b101, 0b10, 0b001101>;
class BNEG_D_ENC : MSA_3R_FMT<0b101, 0b11, 0b001101>;

class BNEGI_B_ENC : MSA_BIT_B_FMT<0b101, 0b001001>;
class BNEGI_H_ENC : MSA_BIT_H_FMT<0b101, 0b001001>;
class BNEGI_W_ENC : MSA_BIT_W_FMT<0b101, 0b001001>;
class BNEGI_D_ENC : MSA_BIT_D_FMT<0b101, 0b001001>;

class BSELI_B_ENC : MSA_I8_FMT<0b10, 0b000001>;

class BSET_B_ENC : MSA_3R_FMT<0b100, 0b00, 0b001101>;
class BSET_H_ENC : MSA_3R_FMT<0b100, 0b01, 0b001101>;
class BSET_W_ENC : MSA_3R_FMT<0b100, 0b10, 0b001101>;
class BSET_D_ENC : MSA_3R_FMT<0b100, 0b11, 0b001101>;

class BSETI_B_ENC : MSA_BIT_B_FMT<0b100, 0b001001>;
class BSETI_H_ENC : MSA_BIT_H_FMT<0b100, 0b001001>;
class BSETI_W_ENC : MSA_BIT_W_FMT<0b100, 0b001001>;
class BSETI_D_ENC : MSA_BIT_D_FMT<0b100, 0b001001>;

class CEQ_B_ENC : MSA_3R_FMT<0b000, 0b00, 0b001111>;
class CEQ_H_ENC : MSA_3R_FMT<0b000, 0b01, 0b001111>;
class CEQ_W_ENC : MSA_3R_FMT<0b000, 0b10, 0b001111>;
class CEQ_D_ENC : MSA_3R_FMT<0b000, 0b11, 0b001111>;

class CEQI_B_ENC : MSA_I5_FMT<0b000, 0b00, 0b000111>;
class CEQI_H_ENC : MSA_I5_FMT<0b000, 0b01, 0b000111>;
class CEQI_W_ENC : MSA_I5_FMT<0b000, 0b10, 0b000111>;
class CEQI_D_ENC : MSA_I5_FMT<0b000, 0b11, 0b000111>;

class CLE_S_B_ENC : MSA_3R_FMT<0b100, 0b00, 0b001111>;
class CLE_S_H_ENC : MSA_3R_FMT<0b100, 0b01, 0b001111>;
class CLE_S_W_ENC : MSA_3R_FMT<0b100, 0b10, 0b001111>;
class CLE_S_D_ENC : MSA_3R_FMT<0b100, 0b11, 0b001111>;

class CLE_U_B_ENC : MSA_3R_FMT<0b101, 0b00, 0b001111>;
class CLE_U_H_ENC : MSA_3R_FMT<0b101, 0b01, 0b001111>;
class CLE_U_W_ENC : MSA_3R_FMT<0b101, 0b10, 0b001111>;
class CLE_U_D_ENC : MSA_3R_FMT<0b101, 0b11, 0b001111>;

class CLEI_S_B_ENC : MSA_I5_FMT<0b100, 0b00, 0b000111>;
class CLEI_S_H_ENC : MSA_I5_FMT<0b100, 0b01, 0b000111>;
class CLEI_S_W_ENC : MSA_I5_FMT<0b100, 0b10, 0b000111>;
class CLEI_S_D_ENC : MSA_I5_FMT<0b100, 0b11, 0b000111>;

class CLEI_U_B_ENC : MSA_I5_FMT<0b101, 0b00, 0b000111>;
class CLEI_U_H_ENC : MSA_I5_FMT<0b101, 0b01, 0b000111>;
class CLEI_U_W_ENC : MSA_I5_FMT<0b101, 0b10, 0b000111>;
class CLEI_U_D_ENC : MSA_I5_FMT<0b101, 0b11, 0b000111>;

class CLT_S_B_ENC : MSA_3R_FMT<0b010, 0b00, 0b001111>;
class CLT_S_H_ENC : MSA_3R_FMT<0b010, 0b01, 0b001111>;
class CLT_S_W_ENC : MSA_3R_FMT<0b010, 0b10, 0b001111>;
class CLT_S_D_ENC : MSA_3R_FMT<0b010, 0b11, 0b001111>;

class CLT_U_B_ENC : MSA_3R_FMT<0b011, 0b00, 0b001111>;
class CLT_U_H_ENC : MSA_3R_FMT<0b011, 0b01, 0b001111>;
class CLT_U_W_ENC : MSA_3R_FMT<0b011, 0b10, 0b001111>;
class CLT_U_D_ENC : MSA_3R_FMT<0b011, 0b11, 0b001111>;

class CLTI_S_B_ENC : MSA_I5_FMT<0b010, 0b00, 0b000111>;
class CLTI_S_H_ENC : MSA_I5_FMT<0b010, 0b01, 0b000111>;
class CLTI_S_W_ENC : MSA_I5_FMT<0b010, 0b10, 0b000111>;
class CLTI_S_D_ENC : MSA_I5_FMT<0b010, 0b11, 0b000111>;

class CLTI_U_B_ENC : MSA_I5_FMT<0b011, 0b00, 0b000111>;
class CLTI_U_H_ENC : MSA_I5_FMT<0b011, 0b01, 0b000111>;
class CLTI_U_W_ENC : MSA_I5_FMT<0b011, 0b10, 0b000111>;
class CLTI_U_D_ENC : MSA_I5_FMT<0b011, 0b11, 0b000111>;

class COPY_S_B_ENC : MSA_ELM_B_FMT<0b0010, 0b011001>;
class COPY_S_H_ENC : MSA_ELM_H_FMT<0b0010, 0b011001>;
class COPY_S_W_ENC : MSA_ELM_W_FMT<0b0010, 0b011001>;

class COPY_U_B_ENC : MSA_ELM_B_FMT<0b0011, 0b011001>;
class COPY_U_H_ENC : MSA_ELM_H_FMT<0b0011, 0b011001>;
class COPY_U_W_ENC : MSA_ELM_W_FMT<0b0011, 0b011001>;

class DIV_S_B_ENC : MSA_3R_FMT<0b100, 0b00, 0b010010>;
class DIV_S_H_ENC : MSA_3R_FMT<0b100, 0b01, 0b010010>;
class DIV_S_W_ENC : MSA_3R_FMT<0b100, 0b10, 0b010010>;
class DIV_S_D_ENC : MSA_3R_FMT<0b100, 0b11, 0b010010>;

class DIV_U_B_ENC : MSA_3R_FMT<0b101, 0b00, 0b010010>;
class DIV_U_H_ENC : MSA_3R_FMT<0b101, 0b01, 0b010010>;
class DIV_U_W_ENC : MSA_3R_FMT<0b101, 0b10, 0b010010>;
class DIV_U_D_ENC : MSA_3R_FMT<0b101, 0b11, 0b010010>;

class DOTP_S_B_ENC : MSA_3R_FMT<0b000, 0b00, 0b010011>;
class DOTP_S_H_ENC : MSA_3R_FMT<0b000, 0b01, 0b010011>;
class DOTP_S_W_ENC : MSA_3R_FMT<0b000, 0b10, 0b010011>;
class DOTP_S_D_ENC : MSA_3R_FMT<0b000, 0b11, 0b010011>;

class DOTP_U_B_ENC : MSA_3R_FMT<0b001, 0b00, 0b010011>;
class DOTP_U_H_ENC : MSA_3R_FMT<0b001, 0b01, 0b010011>;
class DOTP_U_W_ENC : MSA_3R_FMT<0b001, 0b10, 0b010011>;
class DOTP_U_D_ENC : MSA_3R_FMT<0b001, 0b11, 0b010011>;

class DPADD_S_H_ENC : MSA_3R_FMT<0b010, 0b01, 0b010011>;
class DPADD_S_W_ENC : MSA_3R_FMT<0b010, 0b10, 0b010011>;
class DPADD_S_D_ENC : MSA_3R_FMT<0b010, 0b11, 0b010011>;

class DPADD_U_H_ENC : MSA_3R_FMT<0b011, 0b01, 0b010011>;
class DPADD_U_W_ENC : MSA_3R_FMT<0b011, 0b10, 0b010011>;
class DPADD_U_D_ENC : MSA_3R_FMT<0b011, 0b11, 0b010011>;

class DPSUB_S_H_ENC : MSA_3R_FMT<0b100, 0b01, 0b010011>;
class DPSUB_S_W_ENC : MSA_3R_FMT<0b100, 0b10, 0b010011>;
class DPSUB_S_D_ENC : MSA_3R_FMT<0b100, 0b11, 0b010011>;

class DPSUB_U_H_ENC : MSA_3R_FMT<0b101, 0b01, 0b010011>;
class DPSUB_U_W_ENC : MSA_3R_FMT<0b101, 0b10, 0b010011>;
class DPSUB_U_D_ENC : MSA_3R_FMT<0b101, 0b11, 0b010011>;

class ILVEV_B_ENC : MSA_3R_FMT<0b110, 0b00, 0b010100>;
class ILVEV_H_ENC : MSA_3R_FMT<0b110, 0b01, 0b010100>;
class ILVEV_W_ENC : MSA_3R_FMT<0b110, 0b10, 0b010100>;
class ILVEV_D_ENC : MSA_3R_FMT<0b110, 0b11, 0b010100>;

class ILVL_B_ENC : MSA_3R_FMT<0b100, 0b00, 0b010100>;
class ILVL_H_ENC : MSA_3R_FMT<0b100, 0b01, 0b010100>;
class ILVL_W_ENC : MSA_3R_FMT<0b100, 0b10, 0b010100>;
class ILVL_D_ENC : MSA_3R_FMT<0b100, 0b11, 0b010100>;

class ILVOD_B_ENC : MSA_3R_FMT<0b111, 0b00, 0b010100>;
class ILVOD_H_ENC : MSA_3R_FMT<0b111, 0b01, 0b010100>;
class ILVOD_W_ENC : MSA_3R_FMT<0b111, 0b10, 0b010100>;
class ILVOD_D_ENC : MSA_3R_FMT<0b111, 0b11, 0b010100>;

class ILVR_B_ENC : MSA_3R_FMT<0b101, 0b00, 0b010100>;
class ILVR_H_ENC : MSA_3R_FMT<0b101, 0b01, 0b010100>;
class ILVR_W_ENC : MSA_3R_FMT<0b101, 0b10, 0b010100>;
class ILVR_D_ENC : MSA_3R_FMT<0b101, 0b11, 0b010100>;

class INSERT_B_ENC : MSA_ELM_B_FMT<0b0100, 0b011001>;
class INSERT_H_ENC : MSA_ELM_H_FMT<0b0100, 0b011001>;
class INSERT_W_ENC : MSA_ELM_W_FMT<0b0100, 0b011001>;

class LD_B_ENC   : MSA_I5_FMT<0b110, 0b00, 0b000111>;
class LD_H_ENC   : MSA_I5_FMT<0b110, 0b01, 0b000111>;
class LD_W_ENC   : MSA_I5_FMT<0b110, 0b10, 0b000111>;
class LD_D_ENC   : MSA_I5_FMT<0b110, 0b11, 0b000111>;

class LDI_B_ENC   : MSA_I10_FMT<0b010, 0b00, 0b001100>;
class LDI_H_ENC   : MSA_I10_FMT<0b010, 0b01, 0b001100>;
class LDI_W_ENC   : MSA_I10_FMT<0b010, 0b10, 0b001100>;
class LDI_D_ENC   : MSA_I10_FMT<0b010, 0b11, 0b001100>;

class ST_B_ENC   : MSA_I5_FMT<0b111, 0b00, 0b000111>;
class ST_H_ENC   : MSA_I5_FMT<0b111, 0b01, 0b000111>;
class ST_W_ENC   : MSA_I5_FMT<0b111, 0b10, 0b000111>;
class ST_D_ENC   : MSA_I5_FMT<0b111, 0b11, 0b000111>;

// Instruction desc.
class MSA_BIT_D_DESC_BASE<string instr_asm, SDPatternOperator OpNode,
                          InstrItinClass itin, RegisterClass RCWD,
                          RegisterClass RCWS> {
  dag OutOperandList = (outs RCWD:$wd);
  dag InOperandList = (ins RCWS:$ws, uimm6:$u6);
  string AsmString = !strconcat(instr_asm, "\t$wd, $ws, $u6");
  list<dag> Pattern = [(set RCWD:$wd, (OpNode RCWS:$ws, immZExt6:$u6))];
  InstrItinClass Itinerary = itin;
}

class MSA_BIT_W_DESC_BASE<string instr_asm, SDPatternOperator OpNode,
                          InstrItinClass itin, RegisterClass RCWD,
                          RegisterClass RCWS> {
  dag OutOperandList = (outs RCWD:$wd);
  dag InOperandList = (ins RCWS:$ws, uimm5:$u5);
  string AsmString = !strconcat(instr_asm, "\t$wd, $ws, $u5");
  list<dag> Pattern = [(set RCWD:$wd, (OpNode RCWS:$ws, immZExt5:$u5))];
  InstrItinClass Itinerary = itin;
}

class MSA_BIT_H_DESC_BASE<string instr_asm, SDPatternOperator OpNode,
                          InstrItinClass itin, RegisterClass RCWD,
                          RegisterClass RCWS> {
  dag OutOperandList = (outs RCWD:$wd);
  dag InOperandList = (ins RCWS:$ws, uimm4:$u4);
  string AsmString = !strconcat(instr_asm, "\t$wd, $ws, $u4");
  list<dag> Pattern = [(set RCWD:$wd, (OpNode RCWS:$ws, immZExt4:$u4))];
  InstrItinClass Itinerary = itin;
}

class MSA_BIT_B_DESC_BASE<string instr_asm, SDPatternOperator OpNode,
                          InstrItinClass itin, RegisterClass RCWD,
                          RegisterClass RCWS> {
  dag OutOperandList = (outs RCWD:$wd);
  dag InOperandList = (ins RCWS:$ws, uimm3:$u3);
  string AsmString = !strconcat(instr_asm, "\t$wd, $ws, $u3");
  list<dag> Pattern = [(set RCWD:$wd, (OpNode RCWS:$ws, immZExt3:$u3))];
  InstrItinClass Itinerary = itin;
}

class MSA_COPY_DESC_BASE<string instr_asm, SDPatternOperator OpNode,
                         InstrItinClass itin, RegisterClass RCD,
                         RegisterClass RCWS> {
  dag OutOperandList = (outs RCD:$rd);
  dag InOperandList = (ins RCWS:$ws, uimm6:$n);
  string AsmString = !strconcat(instr_asm, "\t$rd, $ws[$n]");
  list<dag> Pattern = [(set RCD:$rd, (OpNode RCWS:$ws, immZExt6:$n))];
  InstrItinClass Itinerary = itin;
}

class MSA_I5_DESC_BASE<string instr_asm, SDPatternOperator OpNode,
                       InstrItinClass itin, RegisterClass RCWD,
                       RegisterClass RCWS> {
  dag OutOperandList = (outs RCWD:$wd);
  dag InOperandList = (ins RCWS:$ws, uimm5:$u5);
  string AsmString = !strconcat(instr_asm, "\t$wd, $ws, $u5");
  list<dag> Pattern = [(set RCWD:$wd, (OpNode RCWS:$ws, immZExt5:$u5))];
  InstrItinClass Itinerary = itin;
}

class MSA_SI5_DESC_BASE<string instr_asm, SDPatternOperator OpNode,
                       InstrItinClass itin, RegisterClass RCWD,
                       RegisterClass RCWS> {
  dag OutOperandList = (outs RCWD:$wd);
  dag InOperandList = (ins RCWS:$ws, simm5:$s5);
  string AsmString = !strconcat(instr_asm, "\t$wd, $ws, $s5");
  list<dag> Pattern = [(set RCWD:$wd, (OpNode RCWS:$ws, immSExt5:$s5))];
  InstrItinClass Itinerary = itin;
}

class MSA_I8_DESC_BASE<string instr_asm, SDPatternOperator OpNode,
                       InstrItinClass itin, RegisterClass RCWD,
                       RegisterClass RCWS> {
  dag OutOperandList = (outs RCWD:$wd);
  dag InOperandList = (ins RCWS:$ws, uimm8:$u8);
  string AsmString = !strconcat(instr_asm, "\t$wd, $ws, $u8");
  list<dag> Pattern = [(set RCWD:$wd, (OpNode RCWS:$ws, immZExt8:$u8))];
  InstrItinClass Itinerary = itin;
}

class MSA_I10_DESC_BASE<string instr_asm, SDPatternOperator OpNode,
                        InstrItinClass itin, RegisterClass RCWD> {
  dag OutOperandList = (outs RCWD:$wd);
  dag InOperandList = (ins simm10:$i10);
  string AsmString = !strconcat(instr_asm, "\t$wd, $i10");
  list<dag> Pattern = [(set RCWD:$wd, (OpNode immSExt10:$i10))];
  InstrItinClass Itinerary = itin;
}

class MSA_2R_DESC_BASE<string instr_asm, SDPatternOperator OpNode,
                       InstrItinClass itin, RegisterClass RCWD,
                       RegisterClass RCWS> {
  dag OutOperandList = (outs RCWD:$wd);
  dag InOperandList = (ins RCWS:$ws);
  string AsmString = !strconcat(instr_asm, "\t$wd, $ws");
  list<dag> Pattern = [(set RCWD:$wd, (OpNode RCWS:$ws))];
  InstrItinClass Itinerary = itin;
}

class MSA_2RF_DESC_BASE<string instr_asm, SDPatternOperator OpNode,
                        InstrItinClass itin, RegisterClass RCWD,
                        RegisterClass RCWS> :
  MSA_2R_DESC_BASE<instr_asm, OpNode, itin, RCWD, RCWS>;


class MSA_3R_DESC_BASE<string instr_asm, SDPatternOperator OpNode,
                       InstrItinClass itin, RegisterClass RCWD,
                       RegisterClass RCWS,  RegisterClass RCWT = RCWS> {
  dag OutOperandList = (outs RCWD:$wd);
  dag InOperandList = (ins RCWS:$ws, RCWT:$wt);
  string AsmString = !strconcat(instr_asm, "\t$wd, $ws, $wt");
  list<dag> Pattern = [(set RCWD:$wd, (OpNode RCWS:$ws, RCWT:$wt))];
  InstrItinClass Itinerary = itin;
}

class MSA_3R_4R_DESC_BASE<string instr_asm, SDPatternOperator OpNode,
                          InstrItinClass itin, RegisterClass RCWD,
                          RegisterClass RCWS,  RegisterClass RCWT = RCWS> {
  dag OutOperandList = (outs RCWD:$wd);
  dag InOperandList = (ins RCWD:$wd_in, RCWS:$ws, RCWT:$wt);
  string AsmString = !strconcat(instr_asm, "\t$wd, $ws, $wt");
  list<dag> Pattern = [(set RCWD:$wd,
                       (OpNode RCWD:$wd_in, RCWS:$ws, RCWT:$wt))];
  InstrItinClass Itinerary = itin;
  string Constraints = "$wd = $wd_in";
}

class MSA_3RF_DESC_BASE<string instr_asm, SDPatternOperator OpNode,
                        InstrItinClass itin, RegisterClass RCWD,
                        RegisterClass RCWS,  RegisterClass RCWT = RCWS> :
  MSA_3R_DESC_BASE<instr_asm, OpNode, itin, RCWD, RCWS, RCWT>;

class MSA_3RF_4RF_DESC_BASE<string instr_asm, SDPatternOperator OpNode,
                            InstrItinClass itin, RegisterClass RCWD,
                            RegisterClass RCWS,  RegisterClass RCWT = RCWS> :
  MSA_3R_4R_DESC_BASE<instr_asm, OpNode, itin, RCWD, RCWS, RCWT>;

class MSA_INSERT_DESC_BASE<string instr_asm, SDPatternOperator OpNode,
                           InstrItinClass itin, RegisterClass RCD,
                           RegisterClass RCWS> {
  dag OutOperandList = (outs RCD:$wd);
  dag InOperandList = (ins RCD:$wd_in, uimm6:$n, RCWS:$rs);
  string AsmString = !strconcat(instr_asm, "\t$wd[$n], $rs");
  list<dag> Pattern = [(set RCD:$wd, (OpNode RCD:$wd_in,
                                             immZExt6:$n,
                                             RCWS:$rs))];
  InstrItinClass Itinerary = itin;
  string Constraints = "$wd = $wd_in";
}

class ADD_A_B_DESC : MSA_3R_DESC_BASE<"add_a.b", int_mips_add_a_b, NoItinerary,
                                      MSA128, MSA128>, IsCommutable;
class ADD_A_H_DESC : MSA_3R_DESC_BASE<"add_a.h", int_mips_add_a_h, NoItinerary,
                                      MSA128, MSA128>, IsCommutable;
class ADD_A_W_DESC : MSA_3R_DESC_BASE<"add_a.w", int_mips_add_a_w, NoItinerary,
                                      MSA128, MSA128>, IsCommutable;
class ADD_A_D_DESC : MSA_3R_DESC_BASE<"add_a.d", int_mips_add_a_d, NoItinerary,
                                      MSA128, MSA128>, IsCommutable;

class ADDS_A_B_DESC : MSA_3R_DESC_BASE<"adds_a.b", int_mips_adds_a_b,
                                       NoItinerary, MSA128, MSA128>,
                                       IsCommutable;
class ADDS_A_H_DESC : MSA_3R_DESC_BASE<"adds_a.h", int_mips_adds_a_h,
                                       NoItinerary, MSA128, MSA128>,
                                       IsCommutable;
class ADDS_A_W_DESC : MSA_3R_DESC_BASE<"adds_a.w", int_mips_adds_a_w,
                                       NoItinerary, MSA128, MSA128>,
                                       IsCommutable;
class ADDS_A_D_DESC : MSA_3R_DESC_BASE<"adds_a.d", int_mips_adds_a_d,
                                       NoItinerary, MSA128, MSA128>,
                                       IsCommutable;

class ADDS_S_B_DESC : MSA_3R_DESC_BASE<"adds_s.b", int_mips_adds_s_b,
                                       NoItinerary, MSA128, MSA128>,
                                       IsCommutable;
class ADDS_S_H_DESC : MSA_3R_DESC_BASE<"adds_s.h", int_mips_adds_s_h,
                                       NoItinerary, MSA128, MSA128>,
                                       IsCommutable;
class ADDS_S_W_DESC : MSA_3R_DESC_BASE<"adds_s.w", int_mips_adds_s_w,
                                       NoItinerary, MSA128, MSA128>,
                                       IsCommutable;
class ADDS_S_D_DESC : MSA_3R_DESC_BASE<"adds_s.d", int_mips_adds_s_d,
                                       NoItinerary, MSA128, MSA128>,
                                       IsCommutable;

class ADDS_U_B_DESC : MSA_3R_DESC_BASE<"adds_u.b", int_mips_adds_u_b,
                                       NoItinerary, MSA128, MSA128>,
                                       IsCommutable;
class ADDS_U_H_DESC : MSA_3R_DESC_BASE<"adds_u.h", int_mips_adds_u_h,
                                       NoItinerary, MSA128, MSA128>,
                                       IsCommutable;
class ADDS_U_W_DESC : MSA_3R_DESC_BASE<"adds_u.w", int_mips_adds_u_w,
                                       NoItinerary, MSA128, MSA128>,
                                       IsCommutable;
class ADDS_U_D_DESC : MSA_3R_DESC_BASE<"adds_u.d", int_mips_adds_u_d,
                                       NoItinerary, MSA128, MSA128>,
                                       IsCommutable;

class ADDV_B_DESC : MSA_3R_DESC_BASE<"addv.b", int_mips_addv_b, NoItinerary,
                                     MSA128, MSA128>, IsCommutable;
class ADDV_H_DESC : MSA_3R_DESC_BASE<"addv.h", int_mips_addv_h, NoItinerary,
                                     MSA128, MSA128>, IsCommutable;
class ADDV_W_DESC : MSA_3R_DESC_BASE<"addv.w", int_mips_addv_w, NoItinerary,
                                     MSA128, MSA128>, IsCommutable;
class ADDV_D_DESC : MSA_3R_DESC_BASE<"addv.d", int_mips_addv_d, NoItinerary,
                                     MSA128, MSA128>, IsCommutable;

class ADDVI_B_DESC : MSA_I5_DESC_BASE<"addvi.b", int_mips_addvi_b, NoItinerary,
                                      MSA128, MSA128>;
class ADDVI_H_DESC : MSA_I5_DESC_BASE<"addvi.h", int_mips_addvi_h, NoItinerary,
                                      MSA128, MSA128>;
class ADDVI_W_DESC : MSA_I5_DESC_BASE<"addvi.w", int_mips_addvi_w, NoItinerary,
                                      MSA128, MSA128>;
class ADDVI_D_DESC : MSA_I5_DESC_BASE<"addvi.d", int_mips_addvi_d, NoItinerary,
                                      MSA128, MSA128>;

class ANDI_B_DESC : MSA_I8_DESC_BASE<"andi.b", int_mips_andi_b, NoItinerary,
                                     MSA128, MSA128>;

class ASUB_S_B_DESC : MSA_3R_DESC_BASE<"asub_s.b", int_mips_asub_s_b,
                                       NoItinerary, MSA128, MSA128>;
class ASUB_S_H_DESC : MSA_3R_DESC_BASE<"asub_s.h", int_mips_asub_s_h,
                                       NoItinerary, MSA128, MSA128>;
class ASUB_S_W_DESC : MSA_3R_DESC_BASE<"asub_s.w", int_mips_asub_s_w,
                                       NoItinerary, MSA128, MSA128>;
class ASUB_S_D_DESC : MSA_3R_DESC_BASE<"asub_s.d", int_mips_asub_s_d,
                                       NoItinerary, MSA128, MSA128>;

class ASUB_U_B_DESC : MSA_3R_DESC_BASE<"asub_u.b", int_mips_asub_u_b,
                                       NoItinerary, MSA128, MSA128>;
class ASUB_U_H_DESC : MSA_3R_DESC_BASE<"asub_u.h", int_mips_asub_u_h,
                                       NoItinerary, MSA128, MSA128>;
class ASUB_U_W_DESC : MSA_3R_DESC_BASE<"asub_u.w", int_mips_asub_u_w,
                                       NoItinerary, MSA128, MSA128>;
class ASUB_U_D_DESC : MSA_3R_DESC_BASE<"asub_u.d", int_mips_asub_u_d,
                                       NoItinerary, MSA128, MSA128>;

class AVE_S_B_DESC : MSA_3R_DESC_BASE<"ave_s.b", int_mips_ave_s_b, NoItinerary,
                                       MSA128, MSA128>, IsCommutable;
class AVE_S_H_DESC : MSA_3R_DESC_BASE<"ave_s.h", int_mips_ave_s_h, NoItinerary,
                                       MSA128, MSA128>, IsCommutable;
class AVE_S_W_DESC : MSA_3R_DESC_BASE<"ave_s.w", int_mips_ave_s_w, NoItinerary,
                                       MSA128, MSA128>, IsCommutable;
class AVE_S_D_DESC : MSA_3R_DESC_BASE<"ave_s.d", int_mips_ave_s_d, NoItinerary,
                                       MSA128, MSA128>, IsCommutable;

class AVE_U_B_DESC : MSA_3R_DESC_BASE<"ave_u.b", int_mips_ave_u_b, NoItinerary,
                                       MSA128, MSA128>, IsCommutable;
class AVE_U_H_DESC : MSA_3R_DESC_BASE<"ave_u.h", int_mips_ave_u_h, NoItinerary,
                                       MSA128, MSA128>, IsCommutable;
class AVE_U_W_DESC : MSA_3R_DESC_BASE<"ave_u.w", int_mips_ave_u_w, NoItinerary,
                                       MSA128, MSA128>, IsCommutable;
class AVE_U_D_DESC : MSA_3R_DESC_BASE<"ave_u.d", int_mips_ave_u_d, NoItinerary,
                                       MSA128, MSA128>, IsCommutable;

class AVER_S_B_DESC : MSA_3R_DESC_BASE<"aver_s.b", int_mips_aver_s_b,
                                       NoItinerary, MSA128, MSA128>,
                                       IsCommutable;
class AVER_S_H_DESC : MSA_3R_DESC_BASE<"aver_s.h", int_mips_aver_s_h,
                                       NoItinerary, MSA128, MSA128>,
                                       IsCommutable;
class AVER_S_W_DESC : MSA_3R_DESC_BASE<"aver_s.w", int_mips_aver_s_w,
                                       NoItinerary, MSA128, MSA128>,
                                       IsCommutable;
class AVER_S_D_DESC : MSA_3R_DESC_BASE<"aver_s.d", int_mips_aver_s_d,
                                       NoItinerary, MSA128, MSA128>,
                                       IsCommutable;

class AVER_U_B_DESC : MSA_3R_DESC_BASE<"aver_u.b", int_mips_aver_u_b,
                                       NoItinerary, MSA128, MSA128>,
                                       IsCommutable;
class AVER_U_H_DESC : MSA_3R_DESC_BASE<"aver_u.h", int_mips_aver_u_h,
                                       NoItinerary, MSA128, MSA128>,
                                       IsCommutable;
class AVER_U_W_DESC : MSA_3R_DESC_BASE<"aver_u.w", int_mips_aver_u_w,
                                       NoItinerary, MSA128, MSA128>,
                                       IsCommutable;
class AVER_U_D_DESC : MSA_3R_DESC_BASE<"aver_u.d", int_mips_aver_u_d,
                                       NoItinerary, MSA128, MSA128>,
                                       IsCommutable;

class BCLR_B_DESC : MSA_3R_DESC_BASE<"bclr.b", int_mips_bclr_b, NoItinerary,
                                     MSA128, MSA128>;
class BCLR_H_DESC : MSA_3R_DESC_BASE<"bclr.h", int_mips_bclr_h, NoItinerary,
                                     MSA128, MSA128>;
class BCLR_W_DESC : MSA_3R_DESC_BASE<"bclr.w", int_mips_bclr_w, NoItinerary,
                                     MSA128, MSA128>;
class BCLR_D_DESC : MSA_3R_DESC_BASE<"bclr.d", int_mips_bclr_d, NoItinerary,
                                     MSA128, MSA128>;

class BCLRI_B_DESC : MSA_BIT_B_DESC_BASE<"bclri.b", int_mips_bclri_b,
                                         NoItinerary, MSA128, MSA128>;
class BCLRI_H_DESC : MSA_BIT_H_DESC_BASE<"bclri.h", int_mips_bclri_h,
                                         NoItinerary, MSA128, MSA128>;
class BCLRI_W_DESC : MSA_BIT_W_DESC_BASE<"bclri.w", int_mips_bclri_w,
                                         NoItinerary, MSA128, MSA128>;
class BCLRI_D_DESC : MSA_BIT_D_DESC_BASE<"bclri.d", int_mips_bclri_d,
                                         NoItinerary, MSA128, MSA128>;

class BINSL_B_DESC : MSA_3R_DESC_BASE<"binsl.b", int_mips_binsl_b, NoItinerary,
                                      MSA128, MSA128>;
class BINSL_H_DESC : MSA_3R_DESC_BASE<"binsl.h", int_mips_binsl_h, NoItinerary,
                                      MSA128, MSA128>;
class BINSL_W_DESC : MSA_3R_DESC_BASE<"binsl.w", int_mips_binsl_w, NoItinerary,
                                      MSA128, MSA128>;
class BINSL_D_DESC : MSA_3R_DESC_BASE<"binsl.d", int_mips_binsl_d, NoItinerary,
                                      MSA128, MSA128>;

class BINSLI_B_DESC : MSA_BIT_B_DESC_BASE<"binsli.b", int_mips_binsli_b,
                                          NoItinerary, MSA128, MSA128>;
class BINSLI_H_DESC : MSA_BIT_H_DESC_BASE<"binsli.h", int_mips_binsli_h,
                                          NoItinerary, MSA128, MSA128>;
class BINSLI_W_DESC : MSA_BIT_W_DESC_BASE<"binsli.w", int_mips_binsli_w,
                                          NoItinerary, MSA128, MSA128>;
class BINSLI_D_DESC : MSA_BIT_D_DESC_BASE<"binsli.d", int_mips_binsli_d,
                                          NoItinerary, MSA128, MSA128>;

class BINSR_B_DESC : MSA_3R_DESC_BASE<"binsr.b", int_mips_binsr_b, NoItinerary,
                                      MSA128, MSA128>;
class BINSR_H_DESC : MSA_3R_DESC_BASE<"binsr.h", int_mips_binsr_h, NoItinerary,
                                      MSA128, MSA128>;
class BINSR_W_DESC : MSA_3R_DESC_BASE<"binsr.w", int_mips_binsr_w, NoItinerary,
                                      MSA128, MSA128>;
class BINSR_D_DESC : MSA_3R_DESC_BASE<"binsr.d", int_mips_binsr_d, NoItinerary,
                                      MSA128, MSA128>;

class BINSRI_B_DESC : MSA_BIT_B_DESC_BASE<"binsri.b", int_mips_binsri_b,
                                          NoItinerary, MSA128, MSA128>;
class BINSRI_H_DESC : MSA_BIT_H_DESC_BASE<"binsri.h", int_mips_binsri_h,
                                          NoItinerary, MSA128, MSA128>;
class BINSRI_W_DESC : MSA_BIT_W_DESC_BASE<"binsri.w", int_mips_binsri_w,
                                          NoItinerary, MSA128, MSA128>;
class BINSRI_D_DESC : MSA_BIT_D_DESC_BASE<"binsri.d", int_mips_binsri_d,
                                          NoItinerary, MSA128, MSA128>;

class BMNZI_B_DESC : MSA_I8_DESC_BASE<"bmnzi.b", int_mips_bmnzi_b, NoItinerary,
                                      MSA128, MSA128>;

class BMZI_B_DESC : MSA_I8_DESC_BASE<"bmzi.b", int_mips_bmzi_b, NoItinerary,
                                     MSA128, MSA128>;

class BNEG_B_DESC : MSA_3R_DESC_BASE<"bneg.b", int_mips_bneg_b, NoItinerary,
                                     MSA128, MSA128>;
class BNEG_H_DESC : MSA_3R_DESC_BASE<"bneg.h", int_mips_bneg_h, NoItinerary,
                                     MSA128, MSA128>;
class BNEG_W_DESC : MSA_3R_DESC_BASE<"bneg.w", int_mips_bneg_w, NoItinerary,
                                     MSA128, MSA128>;
class BNEG_D_DESC : MSA_3R_DESC_BASE<"bneg.d", int_mips_bneg_d, NoItinerary,
                                     MSA128, MSA128>;

class BNEGI_B_DESC : MSA_BIT_B_DESC_BASE<"bnegi.b", int_mips_bnegi_b,
                                         NoItinerary, MSA128, MSA128>;
class BNEGI_H_DESC : MSA_BIT_H_DESC_BASE<"bnegi.h", int_mips_bnegi_h,
                                         NoItinerary, MSA128, MSA128>;
class BNEGI_W_DESC : MSA_BIT_W_DESC_BASE<"bnegi.w", int_mips_bnegi_w,
                                         NoItinerary, MSA128, MSA128>;
class BNEGI_D_DESC : MSA_BIT_D_DESC_BASE<"bnegi.d", int_mips_bnegi_d,
                                         NoItinerary, MSA128, MSA128>;

class BSELI_B_DESC : MSA_I8_DESC_BASE<"bseli.b", int_mips_bseli_b, NoItinerary,
                                      MSA128, MSA128>;

class BSET_B_DESC : MSA_3R_DESC_BASE<"bset.b", int_mips_bset_b, NoItinerary,
                                     MSA128, MSA128>;
class BSET_H_DESC : MSA_3R_DESC_BASE<"bset.h", int_mips_bset_h, NoItinerary,
                                     MSA128, MSA128>;
class BSET_W_DESC : MSA_3R_DESC_BASE<"bset.w", int_mips_bset_w, NoItinerary,
                                     MSA128, MSA128>;
class BSET_D_DESC : MSA_3R_DESC_BASE<"bset.d", int_mips_bset_d, NoItinerary,
                                     MSA128, MSA128>;

class BSETI_B_DESC : MSA_BIT_B_DESC_BASE<"bseti.b", int_mips_bseti_b,
                                         NoItinerary, MSA128, MSA128>;
class BSETI_H_DESC : MSA_BIT_H_DESC_BASE<"bseti.h", int_mips_bseti_h,
                                         NoItinerary, MSA128, MSA128>;
class BSETI_W_DESC : MSA_BIT_W_DESC_BASE<"bseti.w", int_mips_bseti_w,
                                         NoItinerary, MSA128, MSA128>;
class BSETI_D_DESC : MSA_BIT_D_DESC_BASE<"bseti.d", int_mips_bseti_d,
                                         NoItinerary, MSA128, MSA128>;

class CEQ_B_DESC : MSA_3R_DESC_BASE<"ceq.b", int_mips_ceq_b, NoItinerary,
                                    MSA128, MSA128>, IsCommutable;
class CEQ_H_DESC : MSA_3R_DESC_BASE<"ceq.h", int_mips_ceq_h, NoItinerary,
                                    MSA128, MSA128>, IsCommutable;
class CEQ_W_DESC : MSA_3R_DESC_BASE<"ceq.w", int_mips_ceq_w, NoItinerary,
                                    MSA128, MSA128>, IsCommutable;
class CEQ_D_DESC : MSA_3R_DESC_BASE<"ceq.d", int_mips_ceq_d, NoItinerary,
                                    MSA128, MSA128>, IsCommutable;

class CEQI_B_DESC : MSA_SI5_DESC_BASE<"ceqi.b", int_mips_ceqi_b, NoItinerary,
                                      MSA128, MSA128>;
class CEQI_H_DESC : MSA_SI5_DESC_BASE<"ceqi.h", int_mips_ceqi_h, NoItinerary,
                                      MSA128, MSA128>;
class CEQI_W_DESC : MSA_SI5_DESC_BASE<"ceqi.w", int_mips_ceqi_w, NoItinerary,
                                      MSA128, MSA128>;
class CEQI_D_DESC : MSA_SI5_DESC_BASE<"ceqi.d", int_mips_ceqi_d, NoItinerary,
                                      MSA128, MSA128>;

class CLE_S_B_DESC : MSA_3R_DESC_BASE<"cle_s.b", int_mips_cle_s_b, NoItinerary,
                                      MSA128, MSA128>;
class CLE_S_H_DESC : MSA_3R_DESC_BASE<"cle_s.h", int_mips_cle_s_h, NoItinerary,
                                      MSA128, MSA128>;
class CLE_S_W_DESC : MSA_3R_DESC_BASE<"cle_s.w", int_mips_cle_s_w, NoItinerary,
                                      MSA128, MSA128>;
class CLE_S_D_DESC : MSA_3R_DESC_BASE<"cle_s.d", int_mips_cle_s_d, NoItinerary,
                                      MSA128, MSA128>;

class CLE_U_B_DESC : MSA_3R_DESC_BASE<"cle_u.b", int_mips_cle_u_b, NoItinerary,
                                      MSA128, MSA128>;
class CLE_U_H_DESC : MSA_3R_DESC_BASE<"cle_u.h", int_mips_cle_u_h, NoItinerary,
                                      MSA128, MSA128>;
class CLE_U_W_DESC : MSA_3R_DESC_BASE<"cle_u.w", int_mips_cle_u_w, NoItinerary,
                                      MSA128, MSA128>;
class CLE_U_D_DESC : MSA_3R_DESC_BASE<"cle_u.d", int_mips_cle_u_d, NoItinerary,
                                      MSA128, MSA128>;

class CLEI_S_B_DESC : MSA_SI5_DESC_BASE<"clei_s.b", int_mips_clei_s_b,
                                        NoItinerary, MSA128, MSA128>;
class CLEI_S_H_DESC : MSA_SI5_DESC_BASE<"clei_s.h", int_mips_clei_s_h,
                                        NoItinerary, MSA128, MSA128>;
class CLEI_S_W_DESC : MSA_SI5_DESC_BASE<"clei_s.w", int_mips_clei_s_w,
                                        NoItinerary, MSA128, MSA128>;
class CLEI_S_D_DESC : MSA_SI5_DESC_BASE<"clei_s.d", int_mips_clei_s_d,
                                        NoItinerary, MSA128, MSA128>;

class CLEI_U_B_DESC : MSA_SI5_DESC_BASE<"clei_u.b", int_mips_clei_u_b,
                                        NoItinerary, MSA128, MSA128>;
class CLEI_U_H_DESC : MSA_SI5_DESC_BASE<"clei_u.h", int_mips_clei_u_h,
                                        NoItinerary, MSA128, MSA128>;
class CLEI_U_W_DESC : MSA_SI5_DESC_BASE<"clei_u.w", int_mips_clei_u_w,
                                        NoItinerary, MSA128, MSA128>;
class CLEI_U_D_DESC : MSA_SI5_DESC_BASE<"clei_u.d", int_mips_clei_u_d,
                                        NoItinerary, MSA128, MSA128>;

class CLT_S_B_DESC : MSA_3R_DESC_BASE<"clt_s.b", int_mips_clt_s_b, NoItinerary,
                                      MSA128, MSA128>;
class CLT_S_H_DESC : MSA_3R_DESC_BASE<"clt_s.h", int_mips_clt_s_h, NoItinerary,
                                      MSA128, MSA128>;
class CLT_S_W_DESC : MSA_3R_DESC_BASE<"clt_s.w", int_mips_clt_s_w, NoItinerary,
                                      MSA128, MSA128>;
class CLT_S_D_DESC : MSA_3R_DESC_BASE<"clt_s.d", int_mips_clt_s_d, NoItinerary,
                                      MSA128, MSA128>;

class CLT_U_B_DESC : MSA_3R_DESC_BASE<"clt_u.b", int_mips_clt_u_b, NoItinerary,
                                      MSA128, MSA128>;
class CLT_U_H_DESC : MSA_3R_DESC_BASE<"clt_u.h", int_mips_clt_u_h, NoItinerary,
                                      MSA128, MSA128>;
class CLT_U_W_DESC : MSA_3R_DESC_BASE<"clt_u.w", int_mips_clt_u_w, NoItinerary,
                                      MSA128, MSA128>;
class CLT_U_D_DESC : MSA_3R_DESC_BASE<"clt_u.d", int_mips_clt_u_d, NoItinerary,
                                      MSA128, MSA128>;

class CLTI_S_B_DESC : MSA_SI5_DESC_BASE<"clti_s.b", int_mips_clti_s_b,
                                        NoItinerary, MSA128, MSA128>;
class CLTI_S_H_DESC : MSA_SI5_DESC_BASE<"clti_s.h", int_mips_clti_s_h,
                                        NoItinerary, MSA128, MSA128>;
class CLTI_S_W_DESC : MSA_SI5_DESC_BASE<"clti_s.w", int_mips_clti_s_w,
                                        NoItinerary, MSA128, MSA128>;
class CLTI_S_D_DESC : MSA_SI5_DESC_BASE<"clti_s.d", int_mips_clti_s_d,
                                        NoItinerary, MSA128, MSA128>;

class CLTI_U_B_DESC : MSA_SI5_DESC_BASE<"clti_u.b", int_mips_clti_u_b,
                                        NoItinerary, MSA128, MSA128>;
class CLTI_U_H_DESC : MSA_SI5_DESC_BASE<"clti_u.h", int_mips_clti_u_h,
                                        NoItinerary, MSA128, MSA128>;
class CLTI_U_W_DESC : MSA_SI5_DESC_BASE<"clti_u.w", int_mips_clti_u_w,
                                        NoItinerary, MSA128, MSA128>;
class CLTI_U_D_DESC : MSA_SI5_DESC_BASE<"clti_u.d", int_mips_clti_u_d,
                                        NoItinerary, MSA128, MSA128>;

class COPY_S_B_DESC : MSA_COPY_DESC_BASE<"copy_s.b", int_mips_copy_s_b,
                                         NoItinerary, GPR32, MSA128>;
class COPY_S_H_DESC : MSA_COPY_DESC_BASE<"copy_s.h", int_mips_copy_s_h,
                                         NoItinerary, GPR32, MSA128>;
class COPY_S_W_DESC : MSA_COPY_DESC_BASE<"copy_s.w", int_mips_copy_s_w,
                                         NoItinerary, GPR32, MSA128>;

class COPY_U_B_DESC : MSA_COPY_DESC_BASE<"copy_u.b", int_mips_copy_u_b,
                                         NoItinerary, GPR32, MSA128>;
class COPY_U_H_DESC : MSA_COPY_DESC_BASE<"copy_u.h", int_mips_copy_u_h,
                                         NoItinerary, GPR32, MSA128>;
class COPY_U_W_DESC : MSA_COPY_DESC_BASE<"copy_u.w", int_mips_copy_u_w,
                                         NoItinerary, GPR32, MSA128>;

class DIV_S_B_DESC : MSA_3R_DESC_BASE<"div_s.b", int_mips_div_s_b, NoItinerary,
                                      MSA128, MSA128>;
class DIV_S_H_DESC : MSA_3R_DESC_BASE<"div_s.h", int_mips_div_s_h, NoItinerary,
                                      MSA128, MSA128>;
class DIV_S_W_DESC : MSA_3R_DESC_BASE<"div_s.w", int_mips_div_s_w, NoItinerary,
                                      MSA128, MSA128>;
class DIV_S_D_DESC : MSA_3R_DESC_BASE<"div_s.d", int_mips_div_s_d, NoItinerary,
                                      MSA128, MSA128>;

class DIV_U_B_DESC : MSA_3R_DESC_BASE<"div_u.b", int_mips_div_u_b, NoItinerary,
                                      MSA128, MSA128>;
class DIV_U_H_DESC : MSA_3R_DESC_BASE<"div_u.h", int_mips_div_u_h, NoItinerary,
                                      MSA128, MSA128>;
class DIV_U_W_DESC : MSA_3R_DESC_BASE<"div_u.w", int_mips_div_u_w, NoItinerary,
                                      MSA128, MSA128>;
class DIV_U_D_DESC : MSA_3R_DESC_BASE<"div_u.d", int_mips_div_u_d, NoItinerary,
                                      MSA128, MSA128>;

class DOTP_S_B_DESC : MSA_3R_DESC_BASE<"dotp_s.b", int_mips_dotp_s_b,
                                       NoItinerary, MSA128, MSA128>,
                                       IsCommutable;
class DOTP_S_H_DESC : MSA_3R_DESC_BASE<"dotp_s.h", int_mips_dotp_s_h,
                                       NoItinerary, MSA128, MSA128>,
                                       IsCommutable;
class DOTP_S_W_DESC : MSA_3R_DESC_BASE<"dotp_s.w", int_mips_dotp_s_w,
                                       NoItinerary, MSA128, MSA128>,
                                       IsCommutable;
class DOTP_S_D_DESC : MSA_3R_DESC_BASE<"dotp_s.d", int_mips_dotp_s_d,
                                       NoItinerary, MSA128, MSA128>,
                                       IsCommutable;

class DOTP_U_B_DESC : MSA_3R_DESC_BASE<"dotp_u.b", int_mips_dotp_u_b,
                                       NoItinerary, MSA128, MSA128>,
                                       IsCommutable;
class DOTP_U_H_DESC : MSA_3R_DESC_BASE<"dotp_u.h", int_mips_dotp_u_h,
                                       NoItinerary, MSA128, MSA128>,
                                       IsCommutable;
class DOTP_U_W_DESC : MSA_3R_DESC_BASE<"dotp_u.w", int_mips_dotp_u_w,
                                       NoItinerary, MSA128, MSA128>,
                                       IsCommutable;
class DOTP_U_D_DESC : MSA_3R_DESC_BASE<"dotp_u.d", int_mips_dotp_u_d,
                                       NoItinerary, MSA128, MSA128>,
                                       IsCommutable;

class DPADD_S_H_DESC : MSA_3R_4R_DESC_BASE<"dpadd_s.h", int_mips_dpadd_s_h,
                                           NoItinerary, MSA128, MSA128>,
                                           IsCommutable;
class DPADD_S_W_DESC : MSA_3R_4R_DESC_BASE<"dpadd_s.w", int_mips_dpadd_s_w,
                                           NoItinerary, MSA128, MSA128>,
                                           IsCommutable;
class DPADD_S_D_DESC : MSA_3R_4R_DESC_BASE<"dpadd_s.d", int_mips_dpadd_s_d,
                                           NoItinerary, MSA128, MSA128>,
                                           IsCommutable;

class DPADD_U_H_DESC : MSA_3R_4R_DESC_BASE<"dpadd_u.h", int_mips_dpadd_u_h,
                                           NoItinerary, MSA128, MSA128>,
                                           IsCommutable;
class DPADD_U_W_DESC : MSA_3R_4R_DESC_BASE<"dpadd_u.w", int_mips_dpadd_u_w,
                                           NoItinerary, MSA128, MSA128>,
                                           IsCommutable;
class DPADD_U_D_DESC : MSA_3R_4R_DESC_BASE<"dpadd_u.d", int_mips_dpadd_u_d,
                                           NoItinerary, MSA128, MSA128>,
                                           IsCommutable;

class DPSUB_S_H_DESC : MSA_3R_4R_DESC_BASE<"dpsub_s.h", int_mips_dpsub_s_h,
                                           NoItinerary, MSA128, MSA128>;
class DPSUB_S_W_DESC : MSA_3R_4R_DESC_BASE<"dpsub_s.w", int_mips_dpsub_s_w,
                                           NoItinerary, MSA128, MSA128>;
class DPSUB_S_D_DESC : MSA_3R_4R_DESC_BASE<"dpsub_s.d", int_mips_dpsub_s_d,
                                           NoItinerary, MSA128, MSA128>;

class DPSUB_U_H_DESC : MSA_3R_4R_DESC_BASE<"dpsub_u.h", int_mips_dpsub_u_h,
                                           NoItinerary, MSA128, MSA128>;
class DPSUB_U_W_DESC : MSA_3R_4R_DESC_BASE<"dpsub_u.w", int_mips_dpsub_u_w,
                                           NoItinerary, MSA128, MSA128>;
class DPSUB_U_D_DESC : MSA_3R_4R_DESC_BASE<"dpsub_u.d", int_mips_dpsub_u_d,
                                           NoItinerary, MSA128, MSA128>;

class ILVEV_B_DESC : MSA_3R_DESC_BASE<"ilvev.b", int_mips_ilvev_b, NoItinerary,
                                      MSA128, MSA128>;
class ILVEV_H_DESC : MSA_3R_DESC_BASE<"ilvev.h", int_mips_ilvev_h, NoItinerary,
                                      MSA128, MSA128>;
class ILVEV_W_DESC : MSA_3R_DESC_BASE<"ilvev.w", int_mips_ilvev_w, NoItinerary,
                                      MSA128, MSA128>;
class ILVEV_D_DESC : MSA_3R_DESC_BASE<"ilvev.d", int_mips_ilvev_d, NoItinerary,
                                      MSA128, MSA128>;

class ILVL_B_DESC : MSA_3R_DESC_BASE<"ilvl.b", int_mips_ilvl_b, NoItinerary,
                                     MSA128, MSA128>;
class ILVL_H_DESC : MSA_3R_DESC_BASE<"ilvl.h", int_mips_ilvl_h, NoItinerary,
                                     MSA128, MSA128>;
class ILVL_W_DESC : MSA_3R_DESC_BASE<"ilvl.w", int_mips_ilvl_w, NoItinerary,
                                     MSA128, MSA128>;
class ILVL_D_DESC : MSA_3R_DESC_BASE<"ilvl.d", int_mips_ilvl_d, NoItinerary,
                                     MSA128, MSA128>;

class ILVOD_B_DESC : MSA_3R_DESC_BASE<"ilvod.b", int_mips_ilvod_b, NoItinerary,
                                      MSA128, MSA128>;
class ILVOD_H_DESC : MSA_3R_DESC_BASE<"ilvod.h", int_mips_ilvod_h, NoItinerary,
                                      MSA128, MSA128>;
class ILVOD_W_DESC : MSA_3R_DESC_BASE<"ilvod.w", int_mips_ilvod_w, NoItinerary,
                                      MSA128, MSA128>;
class ILVOD_D_DESC : MSA_3R_DESC_BASE<"ilvod.d", int_mips_ilvod_d, NoItinerary,
                                      MSA128, MSA128>;

class ILVR_B_DESC : MSA_3R_DESC_BASE<"ilvr.b", int_mips_ilvr_b, NoItinerary,
                                     MSA128, MSA128>;
class ILVR_H_DESC : MSA_3R_DESC_BASE<"ilvr.h", int_mips_ilvr_h, NoItinerary,
                                     MSA128, MSA128>;
class ILVR_W_DESC : MSA_3R_DESC_BASE<"ilvr.w", int_mips_ilvr_w, NoItinerary,
                                     MSA128, MSA128>;
class ILVR_D_DESC : MSA_3R_DESC_BASE<"ilvr.d", int_mips_ilvr_d, NoItinerary,
                                     MSA128, MSA128>;

class INSERT_B_DESC : MSA_INSERT_DESC_BASE<"insert.b", int_mips_insert_b,
                                           NoItinerary, MSA128, GPR32>;
class INSERT_H_DESC : MSA_INSERT_DESC_BASE<"insert.h", int_mips_insert_h,
                                           NoItinerary, MSA128, GPR32>;
class INSERT_W_DESC : MSA_INSERT_DESC_BASE<"insert.w", int_mips_insert_w,
                                           NoItinerary, MSA128, GPR32>;

class LD_DESC_BASE<string instr_asm, SDPatternOperator OpNode,
                   ValueType TyNode, InstrItinClass itin, RegisterClass RCWD,
                   Operand MemOpnd = mem, ComplexPattern Addr = addr> {
  dag OutOperandList = (outs RCWD:$wd);
  dag InOperandList = (ins MemOpnd:$addr);
  string AsmString = !strconcat(instr_asm, "\t$wd, $addr");
  list<dag> Pattern = [(set RCWD:$wd, (TyNode (OpNode Addr:$addr)))];
  InstrItinClass Itinerary = itin;
}

class LD_B_DESC : LD_DESC_BASE<"ld.b", load, v16i8, NoItinerary, MSA128>;
class LD_H_DESC : LD_DESC_BASE<"ld.h", load, v8i16, NoItinerary, MSA128>;
class LD_W_DESC : LD_DESC_BASE<"ld.w", load, v4i32, NoItinerary, MSA128>;
class LD_D_DESC : LD_DESC_BASE<"ld.d", load, v2i64, NoItinerary, MSA128>;

class LDI_B_DESC : MSA_I10_DESC_BASE<"ldi.b", int_mips_ldi_b,
                                     NoItinerary, MSA128>;
class LDI_H_DESC : MSA_I10_DESC_BASE<"ldi.h", int_mips_ldi_h,
                                     NoItinerary, MSA128>;
class LDI_W_DESC : MSA_I10_DESC_BASE<"ldi.w", int_mips_ldi_w,
                                     NoItinerary, MSA128>;
class LDI_D_DESC : MSA_I10_DESC_BASE<"ldi.d", int_mips_ldi_d,
                                     NoItinerary, MSA128>;

class ST_DESC_BASE<string instr_asm, SDPatternOperator OpNode,
                   ValueType TyNode, InstrItinClass itin, RegisterClass RCWD,
                   Operand MemOpnd = mem, ComplexPattern Addr = addr> {
  dag OutOperandList = (outs);
  dag InOperandList = (ins RCWD:$wd, MemOpnd:$addr);
  string AsmString = !strconcat(instr_asm, "\t$wd, $addr");
  list<dag> Pattern = [(OpNode (TyNode RCWD:$wd), Addr:$addr)];
  InstrItinClass Itinerary = itin;
}

// Load/Store
class ST_B_DESC : ST_DESC_BASE<"st.b", store, v16i8, NoItinerary, MSA128>;
class ST_H_DESC : ST_DESC_BASE<"st.h", store, v8i16, NoItinerary, MSA128>;
class ST_W_DESC : ST_DESC_BASE<"st.w", store, v4i32, NoItinerary, MSA128>;
class ST_D_DESC : ST_DESC_BASE<"st.d", store, v2i64, NoItinerary, MSA128>;

// Instruction defs.
def ADD_A_B : ADD_A_B_ENC, ADD_A_B_DESC, Requires<[HasMSA]>;
def ADD_A_H : ADD_A_H_ENC, ADD_A_H_DESC, Requires<[HasMSA]>;
def ADD_A_W : ADD_A_W_ENC, ADD_A_W_DESC, Requires<[HasMSA]>;
def ADD_A_D : ADD_A_D_ENC, ADD_A_D_DESC, Requires<[HasMSA]>;

def ADDS_A_B : ADDS_A_B_ENC, ADDS_A_B_DESC, Requires<[HasMSA]>;
def ADDS_A_H : ADDS_A_H_ENC, ADDS_A_H_DESC, Requires<[HasMSA]>;
def ADDS_A_W : ADDS_A_W_ENC, ADDS_A_W_DESC, Requires<[HasMSA]>;
def ADDS_A_D : ADDS_A_D_ENC, ADDS_A_D_DESC, Requires<[HasMSA]>;

def ADDS_S_B : ADDS_S_B_ENC, ADDS_S_B_DESC, Requires<[HasMSA]>;
def ADDS_S_H : ADDS_S_H_ENC, ADDS_S_H_DESC, Requires<[HasMSA]>;
def ADDS_S_W : ADDS_S_W_ENC, ADDS_S_W_DESC, Requires<[HasMSA]>;
def ADDS_S_D : ADDS_S_D_ENC, ADDS_S_D_DESC, Requires<[HasMSA]>;

def ADDS_U_B : ADDS_U_B_ENC, ADDS_U_B_DESC, Requires<[HasMSA]>;
def ADDS_U_H : ADDS_U_H_ENC, ADDS_U_H_DESC, Requires<[HasMSA]>;
def ADDS_U_W : ADDS_U_W_ENC, ADDS_U_W_DESC, Requires<[HasMSA]>;
def ADDS_U_D : ADDS_U_D_ENC, ADDS_U_D_DESC, Requires<[HasMSA]>;

def ADDV_B : ADDV_B_ENC, ADDV_B_DESC, Requires<[HasMSA]>;
def ADDV_H : ADDV_H_ENC, ADDV_H_DESC, Requires<[HasMSA]>;
def ADDV_W : ADDV_W_ENC, ADDV_W_DESC, Requires<[HasMSA]>;
def ADDV_D : ADDV_D_ENC, ADDV_D_DESC, Requires<[HasMSA]>;

def ADDVI_B : ADDVI_B_ENC, ADDVI_B_DESC, Requires<[HasMSA]>;
def ADDVI_H : ADDVI_H_ENC, ADDVI_H_DESC, Requires<[HasMSA]>;
def ADDVI_W : ADDVI_W_ENC, ADDVI_W_DESC, Requires<[HasMSA]>;
def ADDVI_D : ADDVI_D_ENC, ADDVI_D_DESC, Requires<[HasMSA]>;

def ANDI_B : ANDI_B_ENC, ANDI_B_DESC, Requires<[HasMSA]>;

def ASUB_S_B : ASUB_S_B_ENC, ASUB_S_B_DESC, Requires<[HasMSA]>;
def ASUB_S_H : ASUB_S_H_ENC, ASUB_S_H_DESC, Requires<[HasMSA]>;
def ASUB_S_W : ASUB_S_W_ENC, ASUB_S_W_DESC, Requires<[HasMSA]>;
def ASUB_S_D : ASUB_S_D_ENC, ASUB_S_D_DESC, Requires<[HasMSA]>;

def ASUB_U_B : ASUB_U_B_ENC, ASUB_U_B_DESC, Requires<[HasMSA]>;
def ASUB_U_H : ASUB_U_H_ENC, ASUB_U_H_DESC, Requires<[HasMSA]>;
def ASUB_U_W : ASUB_U_W_ENC, ASUB_U_W_DESC, Requires<[HasMSA]>;
def ASUB_U_D : ASUB_U_D_ENC, ASUB_U_D_DESC, Requires<[HasMSA]>;

def AVE_S_B : AVE_S_B_ENC, AVE_S_B_DESC, Requires<[HasMSA]>;
def AVE_S_H : AVE_S_H_ENC, AVE_S_H_DESC, Requires<[HasMSA]>;
def AVE_S_W : AVE_S_W_ENC, AVE_S_W_DESC, Requires<[HasMSA]>;
def AVE_S_D : AVE_S_D_ENC, AVE_S_D_DESC, Requires<[HasMSA]>;

def AVE_U_B : AVE_U_B_ENC, AVE_U_B_DESC, Requires<[HasMSA]>;
def AVE_U_H : AVE_U_H_ENC, AVE_U_H_DESC, Requires<[HasMSA]>;
def AVE_U_W : AVE_U_W_ENC, AVE_U_W_DESC, Requires<[HasMSA]>;
def AVE_U_D : AVE_U_D_ENC, AVE_U_D_DESC, Requires<[HasMSA]>;

def AVER_S_B : AVER_S_B_ENC, AVER_S_B_DESC, Requires<[HasMSA]>;
def AVER_S_H : AVER_S_H_ENC, AVER_S_H_DESC, Requires<[HasMSA]>;
def AVER_S_W : AVER_S_W_ENC, AVER_S_W_DESC, Requires<[HasMSA]>;
def AVER_S_D : AVER_S_D_ENC, AVER_S_D_DESC, Requires<[HasMSA]>;

def AVER_U_B : AVER_U_B_ENC, AVER_U_B_DESC, Requires<[HasMSA]>;
def AVER_U_H : AVER_U_H_ENC, AVER_U_H_DESC, Requires<[HasMSA]>;
def AVER_U_W : AVER_U_W_ENC, AVER_U_W_DESC, Requires<[HasMSA]>;
def AVER_U_D : AVER_U_D_ENC, AVER_U_D_DESC, Requires<[HasMSA]>;

def BCLR_B : BCLR_B_ENC, BCLR_B_DESC, Requires<[HasMSA]>;
def BCLR_H : BCLR_H_ENC, BCLR_H_DESC, Requires<[HasMSA]>;
def BCLR_W : BCLR_W_ENC, BCLR_W_DESC, Requires<[HasMSA]>;
def BCLR_D : BCLR_D_ENC, BCLR_D_DESC, Requires<[HasMSA]>;

def BCLRI_B : BCLRI_B_ENC, BCLRI_B_DESC, Requires<[HasMSA]>;
def BCLRI_H : BCLRI_H_ENC, BCLRI_H_DESC, Requires<[HasMSA]>;
def BCLRI_W : BCLRI_W_ENC, BCLRI_W_DESC, Requires<[HasMSA]>;
def BCLRI_D : BCLRI_D_ENC, BCLRI_D_DESC, Requires<[HasMSA]>;

def BINSL_B : BINSL_B_ENC, BINSL_B_DESC, Requires<[HasMSA]>;
def BINSL_H : BINSL_H_ENC, BINSL_H_DESC, Requires<[HasMSA]>;
def BINSL_W : BINSL_W_ENC, BINSL_W_DESC, Requires<[HasMSA]>;
def BINSL_D : BINSL_D_ENC, BINSL_D_DESC, Requires<[HasMSA]>;

def BINSLI_B : BINSLI_B_ENC, BINSLI_B_DESC, Requires<[HasMSA]>;
def BINSLI_H : BINSLI_H_ENC, BINSLI_H_DESC, Requires<[HasMSA]>;
def BINSLI_W : BINSLI_W_ENC, BINSLI_W_DESC, Requires<[HasMSA]>;
def BINSLI_D : BINSLI_D_ENC, BINSLI_D_DESC, Requires<[HasMSA]>;

def BINSR_B : BINSR_B_ENC, BINSR_B_DESC, Requires<[HasMSA]>;
def BINSR_H : BINSR_H_ENC, BINSR_H_DESC, Requires<[HasMSA]>;
def BINSR_W : BINSR_W_ENC, BINSR_W_DESC, Requires<[HasMSA]>;
def BINSR_D : BINSR_D_ENC, BINSR_D_DESC, Requires<[HasMSA]>;

def BINSRI_B : BINSRI_B_ENC, BINSRI_B_DESC, Requires<[HasMSA]>;
def BINSRI_H : BINSRI_H_ENC, BINSRI_H_DESC, Requires<[HasMSA]>;
def BINSRI_W : BINSRI_W_ENC, BINSRI_W_DESC, Requires<[HasMSA]>;
def BINSRI_D : BINSRI_D_ENC, BINSRI_D_DESC, Requires<[HasMSA]>;

def BMNZI_B : BMNZI_B_ENC, BMNZI_B_DESC, Requires<[HasMSA]>;

def BMZI_B : BMZI_B_ENC, BMZI_B_DESC, Requires<[HasMSA]>;

def BNEG_B : BNEG_B_ENC, BNEG_B_DESC, Requires<[HasMSA]>;
def BNEG_H : BNEG_H_ENC, BNEG_H_DESC, Requires<[HasMSA]>;
def BNEG_W : BNEG_W_ENC, BNEG_W_DESC, Requires<[HasMSA]>;
def BNEG_D : BNEG_D_ENC, BNEG_D_DESC, Requires<[HasMSA]>;

def BNEGI_B : BNEGI_B_ENC, BNEGI_B_DESC, Requires<[HasMSA]>;
def BNEGI_H : BNEGI_H_ENC, BNEGI_H_DESC, Requires<[HasMSA]>;
def BNEGI_W : BNEGI_W_ENC, BNEGI_W_DESC, Requires<[HasMSA]>;
def BNEGI_D : BNEGI_D_ENC, BNEGI_D_DESC, Requires<[HasMSA]>;

def BSELI_B : BSELI_B_ENC, BSELI_B_DESC, Requires<[HasMSA]>;

def BSET_B : BSET_B_ENC, BSET_B_DESC, Requires<[HasMSA]>;
def BSET_H : BSET_H_ENC, BSET_H_DESC, Requires<[HasMSA]>;
def BSET_W : BSET_W_ENC, BSET_W_DESC, Requires<[HasMSA]>;
def BSET_D : BSET_D_ENC, BSET_D_DESC, Requires<[HasMSA]>;

def BSETI_B : BSETI_B_ENC, BSETI_B_DESC, Requires<[HasMSA]>;
def BSETI_H : BSETI_H_ENC, BSETI_H_DESC, Requires<[HasMSA]>;
def BSETI_W : BSETI_W_ENC, BSETI_W_DESC, Requires<[HasMSA]>;
def BSETI_D : BSETI_D_ENC, BSETI_D_DESC, Requires<[HasMSA]>;

def CEQ_B : CEQ_B_ENC, CEQ_B_DESC, Requires<[HasMSA]>;
def CEQ_H : CEQ_H_ENC, CEQ_H_DESC, Requires<[HasMSA]>;
def CEQ_W : CEQ_W_ENC, CEQ_W_DESC, Requires<[HasMSA]>;
def CEQ_D : CEQ_D_ENC, CEQ_D_DESC, Requires<[HasMSA]>;

def CEQI_B : CEQI_B_ENC, CEQI_B_DESC, Requires<[HasMSA]>;
def CEQI_H : CEQI_H_ENC, CEQI_H_DESC, Requires<[HasMSA]>;
def CEQI_W : CEQI_W_ENC, CEQI_W_DESC, Requires<[HasMSA]>;
def CEQI_D : CEQI_D_ENC, CEQI_D_DESC, Requires<[HasMSA]>;

def CLE_S_B : CLE_S_B_ENC, CLE_S_B_DESC, Requires<[HasMSA]>;
def CLE_S_H : CLE_S_H_ENC, CLE_S_H_DESC, Requires<[HasMSA]>;
def CLE_S_W : CLE_S_W_ENC, CLE_S_W_DESC, Requires<[HasMSA]>;
def CLE_S_D : CLE_S_D_ENC, CLE_S_D_DESC, Requires<[HasMSA]>;

def CLE_U_B : CLE_U_B_ENC, CLE_U_B_DESC, Requires<[HasMSA]>;
def CLE_U_H : CLE_U_H_ENC, CLE_U_H_DESC, Requires<[HasMSA]>;
def CLE_U_W : CLE_U_W_ENC, CLE_U_W_DESC, Requires<[HasMSA]>;
def CLE_U_D : CLE_U_D_ENC, CLE_U_D_DESC, Requires<[HasMSA]>;

def CLEI_S_B : CLEI_S_B_ENC, CLEI_S_B_DESC, Requires<[HasMSA]>;
def CLEI_S_H : CLEI_S_H_ENC, CLEI_S_H_DESC, Requires<[HasMSA]>;
def CLEI_S_W : CLEI_S_W_ENC, CLEI_S_W_DESC, Requires<[HasMSA]>;
def CLEI_S_D : CLEI_S_D_ENC, CLEI_S_D_DESC, Requires<[HasMSA]>;

def CLEI_U_B : CLEI_U_B_ENC, CLEI_U_B_DESC, Requires<[HasMSA]>;
def CLEI_U_H : CLEI_U_H_ENC, CLEI_U_H_DESC, Requires<[HasMSA]>;
def CLEI_U_W : CLEI_U_W_ENC, CLEI_U_W_DESC, Requires<[HasMSA]>;
def CLEI_U_D : CLEI_U_D_ENC, CLEI_U_D_DESC, Requires<[HasMSA]>;

def CLT_S_B : CLT_S_B_ENC, CLT_S_B_DESC, Requires<[HasMSA]>;
def CLT_S_H : CLT_S_H_ENC, CLT_S_H_DESC, Requires<[HasMSA]>;
def CLT_S_W : CLT_S_W_ENC, CLT_S_W_DESC, Requires<[HasMSA]>;
def CLT_S_D : CLT_S_D_ENC, CLT_S_D_DESC, Requires<[HasMSA]>;

def CLT_U_B : CLT_U_B_ENC, CLT_U_B_DESC, Requires<[HasMSA]>;
def CLT_U_H : CLT_U_H_ENC, CLT_U_H_DESC, Requires<[HasMSA]>;
def CLT_U_W : CLT_U_W_ENC, CLT_U_W_DESC, Requires<[HasMSA]>;
def CLT_U_D : CLT_U_D_ENC, CLT_U_D_DESC, Requires<[HasMSA]>;

def CLTI_S_B : CLTI_S_B_ENC, CLTI_S_B_DESC, Requires<[HasMSA]>;
def CLTI_S_H : CLTI_S_H_ENC, CLTI_S_H_DESC, Requires<[HasMSA]>;
def CLTI_S_W : CLTI_S_W_ENC, CLTI_S_W_DESC, Requires<[HasMSA]>;
def CLTI_S_D : CLTI_S_D_ENC, CLTI_S_D_DESC, Requires<[HasMSA]>;

def CLTI_U_B : CLTI_U_B_ENC, CLTI_U_B_DESC, Requires<[HasMSA]>;
def CLTI_U_H : CLTI_U_H_ENC, CLTI_U_H_DESC, Requires<[HasMSA]>;
def CLTI_U_W : CLTI_U_W_ENC, CLTI_U_W_DESC, Requires<[HasMSA]>;
def CLTI_U_D : CLTI_U_D_ENC, CLTI_U_D_DESC, Requires<[HasMSA]>;

def COPY_S_B : COPY_S_B_ENC, COPY_S_B_DESC, Requires<[HasMSA]>;
def COPY_S_H : COPY_S_H_ENC, COPY_S_H_DESC, Requires<[HasMSA]>;
def COPY_S_W : COPY_S_W_ENC, COPY_S_W_DESC, Requires<[HasMSA]>;

def COPY_U_B : COPY_U_B_ENC, COPY_U_B_DESC, Requires<[HasMSA]>;
def COPY_U_H : COPY_U_H_ENC, COPY_U_H_DESC, Requires<[HasMSA]>;
def COPY_U_W : COPY_U_W_ENC, COPY_U_W_DESC, Requires<[HasMSA]>;

def DIV_S_B : DIV_S_B_ENC, DIV_S_B_DESC, Requires<[HasMSA]>;
def DIV_S_H : DIV_S_H_ENC, DIV_S_H_DESC, Requires<[HasMSA]>;
def DIV_S_W : DIV_S_W_ENC, DIV_S_W_DESC, Requires<[HasMSA]>;
def DIV_S_D : DIV_S_D_ENC, DIV_S_D_DESC, Requires<[HasMSA]>;

def DIV_U_B : DIV_U_B_ENC, DIV_U_B_DESC, Requires<[HasMSA]>;
def DIV_U_H : DIV_U_H_ENC, DIV_U_H_DESC, Requires<[HasMSA]>;
def DIV_U_W : DIV_U_W_ENC, DIV_U_W_DESC, Requires<[HasMSA]>;
def DIV_U_D : DIV_U_D_ENC, DIV_U_D_DESC, Requires<[HasMSA]>;

def DOTP_S_B : DOTP_S_B_ENC, DOTP_S_B_DESC, Requires<[HasMSA]>;
def DOTP_S_H : DOTP_S_H_ENC, DOTP_S_H_DESC, Requires<[HasMSA]>;
def DOTP_S_W : DOTP_S_W_ENC, DOTP_S_W_DESC, Requires<[HasMSA]>;
def DOTP_S_D : DOTP_S_D_ENC, DOTP_S_D_DESC, Requires<[HasMSA]>;

def DOTP_U_B : DOTP_U_B_ENC, DOTP_U_B_DESC, Requires<[HasMSA]>;
def DOTP_U_H : DOTP_U_H_ENC, DOTP_U_H_DESC, Requires<[HasMSA]>;
def DOTP_U_W : DOTP_U_W_ENC, DOTP_U_W_DESC, Requires<[HasMSA]>;
def DOTP_U_D : DOTP_U_D_ENC, DOTP_U_D_DESC, Requires<[HasMSA]>;

def DPADD_S_H : DPADD_S_H_ENC, DPADD_S_H_DESC, Requires<[HasMSA]>;
def DPADD_S_W : DPADD_S_W_ENC, DPADD_S_W_DESC, Requires<[HasMSA]>;
def DPADD_S_D : DPADD_S_D_ENC, DPADD_S_D_DESC, Requires<[HasMSA]>;

def DPADD_U_H : DPADD_U_H_ENC, DPADD_U_H_DESC, Requires<[HasMSA]>;
def DPADD_U_W : DPADD_U_W_ENC, DPADD_U_W_DESC, Requires<[HasMSA]>;
def DPADD_U_D : DPADD_U_D_ENC, DPADD_U_D_DESC, Requires<[HasMSA]>;

def DPSUB_S_H : DPSUB_S_H_ENC, DPSUB_S_H_DESC, Requires<[HasMSA]>;
def DPSUB_S_W : DPSUB_S_W_ENC, DPSUB_S_W_DESC, Requires<[HasMSA]>;
def DPSUB_S_D : DPSUB_S_D_ENC, DPSUB_S_D_DESC, Requires<[HasMSA]>;

def DPSUB_U_H : DPSUB_U_H_ENC, DPSUB_U_H_DESC, Requires<[HasMSA]>;
def DPSUB_U_W : DPSUB_U_W_ENC, DPSUB_U_W_DESC, Requires<[HasMSA]>;
def DPSUB_U_D : DPSUB_U_D_ENC, DPSUB_U_D_DESC, Requires<[HasMSA]>;

def ILVEV_B : ILVEV_B_ENC, ILVEV_B_DESC, Requires<[HasMSA]>;
def ILVEV_H : ILVEV_H_ENC, ILVEV_H_DESC, Requires<[HasMSA]>;
def ILVEV_W : ILVEV_W_ENC, ILVEV_W_DESC, Requires<[HasMSA]>;
def ILVEV_D : ILVEV_D_ENC, ILVEV_D_DESC, Requires<[HasMSA]>;

def ILVL_B : ILVL_B_ENC, ILVL_B_DESC, Requires<[HasMSA]>;
def ILVL_H : ILVL_H_ENC, ILVL_H_DESC, Requires<[HasMSA]>;
def ILVL_W : ILVL_W_ENC, ILVL_W_DESC, Requires<[HasMSA]>;
def ILVL_D : ILVL_D_ENC, ILVL_D_DESC, Requires<[HasMSA]>;

def ILVOD_B : ILVOD_B_ENC, ILVOD_B_DESC, Requires<[HasMSA]>;
def ILVOD_H : ILVOD_H_ENC, ILVOD_H_DESC, Requires<[HasMSA]>;
def ILVOD_W : ILVOD_W_ENC, ILVOD_W_DESC, Requires<[HasMSA]>;
def ILVOD_D : ILVOD_D_ENC, ILVOD_D_DESC, Requires<[HasMSA]>;

def ILVR_B : ILVR_B_ENC, ILVR_B_DESC, Requires<[HasMSA]>;
def ILVR_H : ILVR_H_ENC, ILVR_H_DESC, Requires<[HasMSA]>;
def ILVR_W : ILVR_W_ENC, ILVR_W_DESC, Requires<[HasMSA]>;
def ILVR_D : ILVR_D_ENC, ILVR_D_DESC, Requires<[HasMSA]>;

def INSERT_B : INSERT_B_ENC, INSERT_B_DESC, Requires<[HasMSA]>;
def INSERT_H : INSERT_H_ENC, INSERT_H_DESC, Requires<[HasMSA]>;
def INSERT_W : INSERT_W_ENC, INSERT_W_DESC, Requires<[HasMSA]>;

def LD_B: LD_B_ENC, LD_B_DESC, Requires<[HasMSA]>;
def LD_H: LD_H_ENC, LD_H_DESC, Requires<[HasMSA]>;
def LD_W: LD_W_ENC, LD_W_DESC, Requires<[HasMSA]>;
def LD_D: LD_D_ENC, LD_D_DESC, Requires<[HasMSA]>;

def LDI_B : LDI_B_ENC, LDI_B_DESC, Requires<[HasMSA]>;
def LDI_H : LDI_H_ENC, LDI_H_DESC, Requires<[HasMSA]>;
def LDI_W : LDI_W_ENC, LDI_W_DESC, Requires<[HasMSA]>;

def ST_B: ST_B_ENC, ST_B_DESC, Requires<[HasMSA]>;
def ST_H: ST_H_ENC, ST_H_DESC, Requires<[HasMSA]>;
def ST_W: ST_W_ENC, ST_W_DESC, Requires<[HasMSA]>;
def ST_D: ST_D_ENC, ST_D_DESC, Requires<[HasMSA]>;

// Patterns.
class MSAPat<dag pattern, dag result, Predicate pred = HasMSA> :
  Pat<pattern, result>, Requires<[pred]>;

def LD_FW : MSAPat<(v4f32 (load addr:$addr)),
                   (LD_W addr:$addr)>;
def LD_FD : MSAPat<(v2f64 (load addr:$addr)),
                   (LD_D addr:$addr)>;

def ST_FW : MSAPat<(store (v4f32 MSA128:$ws), addr:$addr),
                   (ST_W MSA128:$ws, addr:$addr)>;
def ST_FD : MSAPat<(store (v2f64 MSA128:$ws), addr:$addr),
                   (ST_D MSA128:$ws, addr:$addr)>;
