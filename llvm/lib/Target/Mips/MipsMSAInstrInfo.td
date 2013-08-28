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

def SDT_MipsVecCond : SDTypeProfile<1, 1, [SDTCisInt<0>, SDTCisVec<1>]>;

def MipsVAllNonZero : SDNode<"MipsISD::VALL_NONZERO", SDT_MipsVecCond>;
def MipsVAnyNonZero : SDNode<"MipsISD::VANY_NONZERO", SDT_MipsVecCond>;
def MipsVAllZero : SDNode<"MipsISD::VALL_ZERO", SDT_MipsVecCond>;
def MipsVAnyZero : SDNode<"MipsISD::VANY_ZERO", SDT_MipsVecCond>;

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

class AND_V_ENC : MSA_VEC_FMT<0b00000, 0b011110>;

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

class BMNZ_V_ENC : MSA_VEC_FMT<0b00100, 0b011110>;

class BMNZI_B_ENC : MSA_I8_FMT<0b00, 0b000001>;

class BMZ_V_ENC : MSA_VEC_FMT<0b00101, 0b011110>;

class BMZI_B_ENC : MSA_I8_FMT<0b01, 0b000001>;

class BNEG_B_ENC : MSA_3R_FMT<0b101, 0b00, 0b001101>;
class BNEG_H_ENC : MSA_3R_FMT<0b101, 0b01, 0b001101>;
class BNEG_W_ENC : MSA_3R_FMT<0b101, 0b10, 0b001101>;
class BNEG_D_ENC : MSA_3R_FMT<0b101, 0b11, 0b001101>;

class BNEGI_B_ENC : MSA_BIT_B_FMT<0b101, 0b001001>;
class BNEGI_H_ENC : MSA_BIT_H_FMT<0b101, 0b001001>;
class BNEGI_W_ENC : MSA_BIT_W_FMT<0b101, 0b001001>;
class BNEGI_D_ENC : MSA_BIT_D_FMT<0b101, 0b001001>;

class BNZ_B_ENC : MSA_I10_FMT<0b000, 0b00, 0b001100>;
class BNZ_H_ENC : MSA_I10_FMT<0b000, 0b01, 0b001100>;
class BNZ_W_ENC : MSA_I10_FMT<0b000, 0b10, 0b001100>;
class BNZ_D_ENC : MSA_I10_FMT<0b000, 0b11, 0b001100>;

class BNZ_V_ENC : MSA_VEC_FMT<0b01000, 0b011110>;

class BSEL_V_ENC : MSA_VECS10_FMT<0b00110, 0b011110>;

class BSELI_B_ENC : MSA_I8_FMT<0b10, 0b000001>;

class BSET_B_ENC : MSA_3R_FMT<0b100, 0b00, 0b001101>;
class BSET_H_ENC : MSA_3R_FMT<0b100, 0b01, 0b001101>;
class BSET_W_ENC : MSA_3R_FMT<0b100, 0b10, 0b001101>;
class BSET_D_ENC : MSA_3R_FMT<0b100, 0b11, 0b001101>;

class BSETI_B_ENC : MSA_BIT_B_FMT<0b100, 0b001001>;
class BSETI_H_ENC : MSA_BIT_H_FMT<0b100, 0b001001>;
class BSETI_W_ENC : MSA_BIT_W_FMT<0b100, 0b001001>;
class BSETI_D_ENC : MSA_BIT_D_FMT<0b100, 0b001001>;

class BZ_B_ENC : MSA_I10_FMT<0b001, 0b00, 0b001100>;
class BZ_H_ENC : MSA_I10_FMT<0b001, 0b01, 0b001100>;
class BZ_W_ENC : MSA_I10_FMT<0b001, 0b10, 0b001100>;
class BZ_D_ENC : MSA_I10_FMT<0b001, 0b11, 0b001100>;

class BZ_V_ENC : MSA_VECS10_FMT<0b01001, 0b011110>;

class CEQ_B_ENC : MSA_3R_FMT<0b000, 0b00, 0b001111>;
class CEQ_H_ENC : MSA_3R_FMT<0b000, 0b01, 0b001111>;
class CEQ_W_ENC : MSA_3R_FMT<0b000, 0b10, 0b001111>;
class CEQ_D_ENC : MSA_3R_FMT<0b000, 0b11, 0b001111>;

class CEQI_B_ENC : MSA_I5_FMT<0b000, 0b00, 0b000111>;
class CEQI_H_ENC : MSA_I5_FMT<0b000, 0b01, 0b000111>;
class CEQI_W_ENC : MSA_I5_FMT<0b000, 0b10, 0b000111>;
class CEQI_D_ENC : MSA_I5_FMT<0b000, 0b11, 0b000111>;

class CFCMSA_ENC : MSA_ELM_FMT<0b0001111110, 0b011001>;

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

class CTCMSA_ENC : MSA_ELM_FMT<0b0000111110, 0b011001>;

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

class FADD_W_ENC : MSA_3RF_FMT<0b0000, 0b0, 0b011011>;
class FADD_D_ENC : MSA_3RF_FMT<0b0000, 0b1, 0b011011>;

class FCAF_W_ENC : MSA_3RF_FMT<0b0000, 0b0, 0b011010>;
class FCAF_D_ENC : MSA_3RF_FMT<0b0000, 0b1, 0b011010>;

class FCEQ_W_ENC : MSA_3RF_FMT<0b0010, 0b0, 0b011010>;
class FCEQ_D_ENC : MSA_3RF_FMT<0b0010, 0b1, 0b011010>;

class FCLASS_W_ENC : MSA_2RF_FMT<0b110010000, 0b0, 0b011110>;
class FCLASS_D_ENC : MSA_2RF_FMT<0b110010000, 0b1, 0b011110>;

class FCLE_W_ENC : MSA_3RF_FMT<0b0110, 0b0, 0b011010>;
class FCLE_D_ENC : MSA_3RF_FMT<0b0110, 0b1, 0b011010>;

class FCLT_W_ENC : MSA_3RF_FMT<0b0100, 0b0, 0b011010>;
class FCLT_D_ENC : MSA_3RF_FMT<0b0100, 0b1, 0b011010>;

class FCNE_W_ENC : MSA_3RF_FMT<0b0011, 0b0, 0b011100>;
class FCNE_D_ENC : MSA_3RF_FMT<0b0011, 0b1, 0b011100>;

class FCOR_W_ENC : MSA_3RF_FMT<0b0001, 0b0, 0b011100>;
class FCOR_D_ENC : MSA_3RF_FMT<0b0001, 0b1, 0b011100>;

class FCUEQ_W_ENC : MSA_3RF_FMT<0b0011, 0b0, 0b011010>;
class FCUEQ_D_ENC : MSA_3RF_FMT<0b0011, 0b1, 0b011010>;

class FCULE_W_ENC : MSA_3RF_FMT<0b0111, 0b0, 0b011010>;
class FCULE_D_ENC : MSA_3RF_FMT<0b0111, 0b1, 0b011010>;

class FCULT_W_ENC : MSA_3RF_FMT<0b0101, 0b0, 0b011010>;
class FCULT_D_ENC : MSA_3RF_FMT<0b0101, 0b1, 0b011010>;

class FCUN_W_ENC : MSA_3RF_FMT<0b0001, 0b0, 0b011010>;
class FCUN_D_ENC : MSA_3RF_FMT<0b0001, 0b1, 0b011010>;

class FCUNE_W_ENC : MSA_3RF_FMT<0b0010, 0b0, 0b011100>;
class FCUNE_D_ENC : MSA_3RF_FMT<0b0010, 0b1, 0b011100>;

class FDIV_W_ENC : MSA_3RF_FMT<0b0011, 0b0, 0b011011>;
class FDIV_D_ENC : MSA_3RF_FMT<0b0011, 0b1, 0b011011>;

class FEXDO_H_ENC : MSA_3RF_FMT<0b1000, 0b0, 0b011011>;
class FEXDO_W_ENC : MSA_3RF_FMT<0b1000, 0b1, 0b011011>;

class FEXP2_W_ENC : MSA_3RF_FMT<0b0111, 0b0, 0b011011>;
class FEXP2_D_ENC : MSA_3RF_FMT<0b0111, 0b1, 0b011011>;

class FEXUPL_W_ENC : MSA_2RF_FMT<0b110011000, 0b0, 0b011110>;
class FEXUPL_D_ENC : MSA_2RF_FMT<0b110011000, 0b1, 0b011110>;

class FEXUPR_W_ENC : MSA_2RF_FMT<0b110011001, 0b0, 0b011110>;
class FEXUPR_D_ENC : MSA_2RF_FMT<0b110011001, 0b1, 0b011110>;

class FFINT_S_W_ENC : MSA_2RF_FMT<0b110011110, 0b0, 0b011110>;
class FFINT_S_D_ENC : MSA_2RF_FMT<0b110011110, 0b1, 0b011110>;

class FFINT_U_W_ENC : MSA_2RF_FMT<0b110011111, 0b0, 0b011110>;
class FFINT_U_D_ENC : MSA_2RF_FMT<0b110011111, 0b1, 0b011110>;

class FFQL_W_ENC : MSA_2RF_FMT<0b110011010, 0b0, 0b011110>;
class FFQL_D_ENC : MSA_2RF_FMT<0b110011010, 0b1, 0b011110>;

class FFQR_W_ENC : MSA_2RF_FMT<0b110011011, 0b0, 0b011110>;
class FFQR_D_ENC : MSA_2RF_FMT<0b110011011, 0b1, 0b011110>;

class FILL_B_ENC : MSA_2R_FMT<0b11000000, 0b00, 0b011110>;
class FILL_H_ENC : MSA_2R_FMT<0b11000000, 0b01, 0b011110>;
class FILL_W_ENC : MSA_2R_FMT<0b11000000, 0b10, 0b011110>;

class FLOG2_W_ENC : MSA_2RF_FMT<0b110010111, 0b0, 0b011110>;
class FLOG2_D_ENC : MSA_2RF_FMT<0b110010111, 0b1, 0b011110>;

class FMADD_W_ENC : MSA_3RF_FMT<0b0100, 0b0, 0b011011>;
class FMADD_D_ENC : MSA_3RF_FMT<0b0100, 0b1, 0b011011>;

class FMAX_W_ENC : MSA_3RF_FMT<0b1110, 0b0, 0b011011>;
class FMAX_D_ENC : MSA_3RF_FMT<0b1110, 0b1, 0b011011>;

class FMAX_A_W_ENC : MSA_3RF_FMT<0b1111, 0b0, 0b011011>;
class FMAX_A_D_ENC : MSA_3RF_FMT<0b1111, 0b1, 0b011011>;

class FMIN_W_ENC : MSA_3RF_FMT<0b1100, 0b0, 0b011011>;
class FMIN_D_ENC : MSA_3RF_FMT<0b1100, 0b1, 0b011011>;

class FMIN_A_W_ENC : MSA_3RF_FMT<0b1101, 0b0, 0b011011>;
class FMIN_A_D_ENC : MSA_3RF_FMT<0b1101, 0b1, 0b011011>;

class FMSUB_W_ENC : MSA_3RF_FMT<0b0101, 0b0, 0b011011>;
class FMSUB_D_ENC : MSA_3RF_FMT<0b0101, 0b1, 0b011011>;

class FMUL_W_ENC : MSA_3RF_FMT<0b0010, 0b0, 0b011011>;
class FMUL_D_ENC : MSA_3RF_FMT<0b0010, 0b1, 0b011011>;

class FRINT_W_ENC : MSA_2RF_FMT<0b110010110, 0b0, 0b011110>;
class FRINT_D_ENC : MSA_2RF_FMT<0b110010110, 0b1, 0b011110>;

class FRCP_W_ENC : MSA_2RF_FMT<0b110010101, 0b0, 0b011110>;
class FRCP_D_ENC : MSA_2RF_FMT<0b110010101, 0b1, 0b011110>;

class FRSQRT_W_ENC : MSA_2RF_FMT<0b110010100, 0b0, 0b011110>;
class FRSQRT_D_ENC : MSA_2RF_FMT<0b110010100, 0b1, 0b011110>;

class FSAF_W_ENC : MSA_3RF_FMT<0b1000, 0b0, 0b011010>;
class FSAF_D_ENC : MSA_3RF_FMT<0b1000, 0b1, 0b011010>;

class FSEQ_W_ENC : MSA_3RF_FMT<0b1010, 0b0, 0b011010>;
class FSEQ_D_ENC : MSA_3RF_FMT<0b1010, 0b1, 0b011010>;

class FSLE_W_ENC : MSA_3RF_FMT<0b1110, 0b0, 0b011010>;
class FSLE_D_ENC : MSA_3RF_FMT<0b1110, 0b1, 0b011010>;

class FSLT_W_ENC : MSA_3RF_FMT<0b1100, 0b0, 0b011010>;
class FSLT_D_ENC : MSA_3RF_FMT<0b1100, 0b1, 0b011010>;

class FSNE_W_ENC : MSA_3RF_FMT<0b1011, 0b0, 0b011100>;
class FSNE_D_ENC : MSA_3RF_FMT<0b1011, 0b1, 0b011100>;

class FSOR_W_ENC : MSA_3RF_FMT<0b1001, 0b0, 0b011100>;
class FSOR_D_ENC : MSA_3RF_FMT<0b1001, 0b1, 0b011100>;

class FSQRT_W_ENC : MSA_2RF_FMT<0b110010011, 0b0, 0b011110>;
class FSQRT_D_ENC : MSA_2RF_FMT<0b110010011, 0b1, 0b011110>;

class FSUB_W_ENC : MSA_3RF_FMT<0b0001, 0b0, 0b011011>;
class FSUB_D_ENC : MSA_3RF_FMT<0b0001, 0b1, 0b011011>;

class FSUEQ_W_ENC : MSA_3RF_FMT<0b1011, 0b0, 0b011010>;
class FSUEQ_D_ENC : MSA_3RF_FMT<0b1011, 0b1, 0b011010>;

class FSULE_W_ENC : MSA_3RF_FMT<0b1111, 0b0, 0b011010>;
class FSULE_D_ENC : MSA_3RF_FMT<0b1111, 0b1, 0b011010>;

class FSULT_W_ENC : MSA_3RF_FMT<0b1101, 0b0, 0b011010>;
class FSULT_D_ENC : MSA_3RF_FMT<0b1101, 0b1, 0b011010>;

class FSUN_W_ENC : MSA_3RF_FMT<0b1001, 0b0, 0b011010>;
class FSUN_D_ENC : MSA_3RF_FMT<0b1001, 0b1, 0b011010>;

class FSUNE_W_ENC : MSA_3RF_FMT<0b1010, 0b0, 0b011100>;
class FSUNE_D_ENC : MSA_3RF_FMT<0b1010, 0b1, 0b011100>;

class FTRUNC_S_W_ENC : MSA_2RF_FMT<0b110100000, 0b0, 0b011110>;
class FTRUNC_S_D_ENC : MSA_2RF_FMT<0b110100000, 0b1, 0b011110>;

class FTRUNC_U_W_ENC : MSA_2RF_FMT<0b110100001, 0b0, 0b011110>;
class FTRUNC_U_D_ENC : MSA_2RF_FMT<0b110100001, 0b1, 0b011110>;

class FTINT_S_W_ENC : MSA_2RF_FMT<0b110011100, 0b0, 0b011110>;
class FTINT_S_D_ENC : MSA_2RF_FMT<0b110011100, 0b1, 0b011110>;

class FTINT_U_W_ENC : MSA_2RF_FMT<0b110011101, 0b0, 0b011110>;
class FTINT_U_D_ENC : MSA_2RF_FMT<0b110011101, 0b1, 0b011110>;

class FTQ_H_ENC : MSA_3RF_FMT<0b1010, 0b0, 0b011011>;
class FTQ_W_ENC : MSA_3RF_FMT<0b1010, 0b1, 0b011011>;

class HADD_S_H_ENC : MSA_3R_FMT<0b100, 0b01, 0b010101>;
class HADD_S_W_ENC : MSA_3R_FMT<0b100, 0b10, 0b010101>;
class HADD_S_D_ENC : MSA_3R_FMT<0b100, 0b11, 0b010101>;

class HADD_U_H_ENC : MSA_3R_FMT<0b101, 0b01, 0b010101>;
class HADD_U_W_ENC : MSA_3R_FMT<0b101, 0b10, 0b010101>;
class HADD_U_D_ENC : MSA_3R_FMT<0b101, 0b11, 0b010101>;

class HSUB_S_H_ENC : MSA_3R_FMT<0b110, 0b01, 0b010101>;
class HSUB_S_W_ENC : MSA_3R_FMT<0b110, 0b10, 0b010101>;
class HSUB_S_D_ENC : MSA_3R_FMT<0b110, 0b11, 0b010101>;

class HSUB_U_H_ENC : MSA_3R_FMT<0b111, 0b01, 0b010101>;
class HSUB_U_W_ENC : MSA_3R_FMT<0b111, 0b10, 0b010101>;
class HSUB_U_D_ENC : MSA_3R_FMT<0b111, 0b11, 0b010101>;

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

class INSVE_B_ENC : MSA_ELM_B_FMT<0b0101, 0b011001>;
class INSVE_H_ENC : MSA_ELM_H_FMT<0b0101, 0b011001>;
class INSVE_W_ENC : MSA_ELM_W_FMT<0b0101, 0b011001>;
class INSVE_D_ENC : MSA_ELM_D_FMT<0b0101, 0b011001>;

class LD_B_ENC   : MSA_I5_FMT<0b110, 0b00, 0b000111>;
class LD_H_ENC   : MSA_I5_FMT<0b110, 0b01, 0b000111>;
class LD_W_ENC   : MSA_I5_FMT<0b110, 0b10, 0b000111>;
class LD_D_ENC   : MSA_I5_FMT<0b110, 0b11, 0b000111>;

class LDI_B_ENC  : MSA_I10_FMT<0b010, 0b00, 0b001100>;
class LDI_H_ENC  : MSA_I10_FMT<0b010, 0b01, 0b001100>;
class LDI_W_ENC  : MSA_I10_FMT<0b010, 0b10, 0b001100>;
class LDI_D_ENC  : MSA_I10_FMT<0b010, 0b11, 0b001100>;

class LDX_B_ENC  : MSA_3R_FMT<0b110, 0b00, 0b001111>;
class LDX_H_ENC  : MSA_3R_FMT<0b110, 0b01, 0b001111>;
class LDX_W_ENC  : MSA_3R_FMT<0b110, 0b10, 0b001111>;
class LDX_D_ENC  : MSA_3R_FMT<0b110, 0b11, 0b001111>;

class MADD_Q_H_ENC : MSA_3RF_FMT<0b0101, 0b0, 0b011100>;
class MADD_Q_W_ENC : MSA_3RF_FMT<0b0101, 0b1, 0b011100>;

class MADDR_Q_H_ENC : MSA_3RF_FMT<0b1101, 0b0, 0b011100>;
class MADDR_Q_W_ENC : MSA_3RF_FMT<0b1101, 0b1, 0b011100>;

class MADDV_B_ENC : MSA_3R_FMT<0b001, 0b00, 0b010010>;
class MADDV_H_ENC : MSA_3R_FMT<0b001, 0b01, 0b010010>;
class MADDV_W_ENC : MSA_3R_FMT<0b001, 0b10, 0b010010>;
class MADDV_D_ENC : MSA_3R_FMT<0b001, 0b11, 0b010010>;

class MAX_A_B_ENC : MSA_3R_FMT<0b110, 0b00, 0b001110>;
class MAX_A_H_ENC : MSA_3R_FMT<0b110, 0b01, 0b001110>;
class MAX_A_W_ENC : MSA_3R_FMT<0b110, 0b10, 0b001110>;
class MAX_A_D_ENC : MSA_3R_FMT<0b110, 0b11, 0b001110>;

class MAX_S_B_ENC : MSA_3R_FMT<0b010, 0b00, 0b001110>;
class MAX_S_H_ENC : MSA_3R_FMT<0b010, 0b01, 0b001110>;
class MAX_S_W_ENC : MSA_3R_FMT<0b010, 0b10, 0b001110>;
class MAX_S_D_ENC : MSA_3R_FMT<0b010, 0b11, 0b001110>;

class MAX_U_B_ENC : MSA_3R_FMT<0b011, 0b00, 0b001110>;
class MAX_U_H_ENC : MSA_3R_FMT<0b011, 0b01, 0b001110>;
class MAX_U_W_ENC : MSA_3R_FMT<0b011, 0b10, 0b001110>;
class MAX_U_D_ENC : MSA_3R_FMT<0b011, 0b11, 0b001110>;

class MAXI_S_B_ENC : MSA_I5_FMT<0b010, 0b00, 0b000110>;
class MAXI_S_H_ENC : MSA_I5_FMT<0b010, 0b01, 0b000110>;
class MAXI_S_W_ENC : MSA_I5_FMT<0b010, 0b10, 0b000110>;
class MAXI_S_D_ENC : MSA_I5_FMT<0b010, 0b11, 0b000110>;

class MAXI_U_B_ENC : MSA_I5_FMT<0b011, 0b00, 0b000110>;
class MAXI_U_H_ENC : MSA_I5_FMT<0b011, 0b01, 0b000110>;
class MAXI_U_W_ENC : MSA_I5_FMT<0b011, 0b10, 0b000110>;
class MAXI_U_D_ENC : MSA_I5_FMT<0b011, 0b11, 0b000110>;

class MIN_A_B_ENC : MSA_3R_FMT<0b111, 0b00, 0b001110>;
class MIN_A_H_ENC : MSA_3R_FMT<0b111, 0b01, 0b001110>;
class MIN_A_W_ENC : MSA_3R_FMT<0b111, 0b10, 0b001110>;
class MIN_A_D_ENC : MSA_3R_FMT<0b111, 0b11, 0b001110>;

class MIN_S_B_ENC : MSA_3R_FMT<0b100, 0b00, 0b001110>;
class MIN_S_H_ENC : MSA_3R_FMT<0b100, 0b01, 0b001110>;
class MIN_S_W_ENC : MSA_3R_FMT<0b100, 0b10, 0b001110>;
class MIN_S_D_ENC : MSA_3R_FMT<0b100, 0b11, 0b001110>;

class MIN_U_B_ENC : MSA_3R_FMT<0b101, 0b00, 0b001110>;
class MIN_U_H_ENC : MSA_3R_FMT<0b101, 0b01, 0b001110>;
class MIN_U_W_ENC : MSA_3R_FMT<0b101, 0b10, 0b001110>;
class MIN_U_D_ENC : MSA_3R_FMT<0b101, 0b11, 0b001110>;

class MINI_S_B_ENC : MSA_I5_FMT<0b100, 0b00, 0b000110>;
class MINI_S_H_ENC : MSA_I5_FMT<0b100, 0b01, 0b000110>;
class MINI_S_W_ENC : MSA_I5_FMT<0b100, 0b10, 0b000110>;
class MINI_S_D_ENC : MSA_I5_FMT<0b100, 0b11, 0b000110>;

class MINI_U_B_ENC : MSA_I5_FMT<0b101, 0b00, 0b000110>;
class MINI_U_H_ENC : MSA_I5_FMT<0b101, 0b01, 0b000110>;
class MINI_U_W_ENC : MSA_I5_FMT<0b101, 0b10, 0b000110>;
class MINI_U_D_ENC : MSA_I5_FMT<0b101, 0b11, 0b000110>;

class MOD_S_B_ENC : MSA_3R_FMT<0b110, 0b00, 0b010010>;
class MOD_S_H_ENC : MSA_3R_FMT<0b110, 0b01, 0b010010>;
class MOD_S_W_ENC : MSA_3R_FMT<0b110, 0b10, 0b010010>;
class MOD_S_D_ENC : MSA_3R_FMT<0b110, 0b11, 0b010010>;

class MOD_U_B_ENC : MSA_3R_FMT<0b111, 0b00, 0b010010>;
class MOD_U_H_ENC : MSA_3R_FMT<0b111, 0b01, 0b010010>;
class MOD_U_W_ENC : MSA_3R_FMT<0b111, 0b10, 0b010010>;
class MOD_U_D_ENC : MSA_3R_FMT<0b111, 0b11, 0b010010>;

class MOVE_V_ENC : MSA_ELM_FMT<0b0010111110, 0b011001>;

class MSUB_Q_H_ENC : MSA_3RF_FMT<0b0110, 0b0, 0b011100>;
class MSUB_Q_W_ENC : MSA_3RF_FMT<0b0110, 0b1, 0b011100>;

class MSUBR_Q_H_ENC : MSA_3RF_FMT<0b1110, 0b0, 0b011100>;
class MSUBR_Q_W_ENC : MSA_3RF_FMT<0b1110, 0b1, 0b011100>;

class MSUBV_B_ENC : MSA_3R_FMT<0b010, 0b00, 0b010010>;
class MSUBV_H_ENC : MSA_3R_FMT<0b010, 0b01, 0b010010>;
class MSUBV_W_ENC : MSA_3R_FMT<0b010, 0b10, 0b010010>;
class MSUBV_D_ENC : MSA_3R_FMT<0b010, 0b11, 0b010010>;

class MUL_Q_H_ENC : MSA_3RF_FMT<0b0000, 0b0, 0b011100>;
class MUL_Q_W_ENC : MSA_3RF_FMT<0b0000, 0b1, 0b011100>;

class MULR_Q_H_ENC : MSA_3RF_FMT<0b1100, 0b0, 0b011100>;
class MULR_Q_W_ENC : MSA_3RF_FMT<0b1100, 0b1, 0b011100>;

class MULV_B_ENC : MSA_3R_FMT<0b000, 0b00, 0b010010>;
class MULV_H_ENC : MSA_3R_FMT<0b000, 0b01, 0b010010>;
class MULV_W_ENC : MSA_3R_FMT<0b000, 0b10, 0b010010>;
class MULV_D_ENC : MSA_3R_FMT<0b000, 0b11, 0b010010>;

class NLOC_B_ENC : MSA_2R_FMT<0b11000010, 0b00, 0b011110>;
class NLOC_H_ENC : MSA_2R_FMT<0b11000010, 0b01, 0b011110>;
class NLOC_W_ENC : MSA_2R_FMT<0b11000010, 0b10, 0b011110>;
class NLOC_D_ENC : MSA_2R_FMT<0b11000010, 0b11, 0b011110>;

class NLZC_B_ENC : MSA_2R_FMT<0b11000011, 0b00, 0b011110>;
class NLZC_H_ENC : MSA_2R_FMT<0b11000011, 0b01, 0b011110>;
class NLZC_W_ENC : MSA_2R_FMT<0b11000011, 0b10, 0b011110>;
class NLZC_D_ENC : MSA_2R_FMT<0b11000011, 0b11, 0b011110>;

class NOR_V_ENC : MSA_VEC_FMT<0b00010, 0b011110>;

class NORI_B_ENC : MSA_I8_FMT<0b10, 0b000000>;

class OR_V_ENC : MSA_VEC_FMT<0b00001, 0b011110>;

class ORI_B_ENC  : MSA_I8_FMT<0b01, 0b000000>;

class PCKEV_B_ENC : MSA_3R_FMT<0b010, 0b00, 0b010100>;
class PCKEV_H_ENC : MSA_3R_FMT<0b010, 0b01, 0b010100>;
class PCKEV_W_ENC : MSA_3R_FMT<0b010, 0b10, 0b010100>;
class PCKEV_D_ENC : MSA_3R_FMT<0b010, 0b11, 0b010100>;

class PCKOD_B_ENC : MSA_3R_FMT<0b011, 0b00, 0b010100>;
class PCKOD_H_ENC : MSA_3R_FMT<0b011, 0b01, 0b010100>;
class PCKOD_W_ENC : MSA_3R_FMT<0b011, 0b10, 0b010100>;
class PCKOD_D_ENC : MSA_3R_FMT<0b011, 0b11, 0b010100>;

class PCNT_B_ENC : MSA_2R_FMT<0b11000001, 0b00, 0b011110>;
class PCNT_H_ENC : MSA_2R_FMT<0b11000001, 0b01, 0b011110>;
class PCNT_W_ENC : MSA_2R_FMT<0b11000001, 0b10, 0b011110>;
class PCNT_D_ENC : MSA_2R_FMT<0b11000001, 0b11, 0b011110>;

class SAT_S_B_ENC : MSA_BIT_B_FMT<0b000, 0b001010>;
class SAT_S_H_ENC : MSA_BIT_H_FMT<0b000, 0b001010>;
class SAT_S_W_ENC : MSA_BIT_W_FMT<0b000, 0b001010>;
class SAT_S_D_ENC : MSA_BIT_D_FMT<0b000, 0b001010>;

class SAT_U_B_ENC : MSA_BIT_B_FMT<0b001, 0b001010>;
class SAT_U_H_ENC : MSA_BIT_H_FMT<0b001, 0b001010>;
class SAT_U_W_ENC : MSA_BIT_W_FMT<0b001, 0b001010>;
class SAT_U_D_ENC : MSA_BIT_D_FMT<0b001, 0b001010>;

class SHF_B_ENC  : MSA_I8_FMT<0b00, 0b000010>;
class SHF_H_ENC  : MSA_I8_FMT<0b01, 0b000010>;
class SHF_W_ENC  : MSA_I8_FMT<0b10, 0b000010>;

class SLD_B_ENC : MSA_3R_FMT<0b000, 0b00, 0b010100>;
class SLD_H_ENC : MSA_3R_FMT<0b000, 0b01, 0b010100>;
class SLD_W_ENC : MSA_3R_FMT<0b000, 0b10, 0b010100>;
class SLD_D_ENC : MSA_3R_FMT<0b000, 0b11, 0b010100>;

class SLDI_B_ENC : MSA_ELM_B_FMT<0b0000, 0b011001>;
class SLDI_H_ENC : MSA_ELM_H_FMT<0b0000, 0b011001>;
class SLDI_W_ENC : MSA_ELM_W_FMT<0b0000, 0b011001>;
class SLDI_D_ENC : MSA_ELM_D_FMT<0b0000, 0b011001>;

class SLL_B_ENC : MSA_3R_FMT<0b000, 0b00, 0b001101>;
class SLL_H_ENC : MSA_3R_FMT<0b000, 0b01, 0b001101>;
class SLL_W_ENC : MSA_3R_FMT<0b000, 0b10, 0b001101>;
class SLL_D_ENC : MSA_3R_FMT<0b000, 0b11, 0b001101>;

class SLLI_B_ENC : MSA_BIT_B_FMT<0b000, 0b001001>;
class SLLI_H_ENC : MSA_BIT_H_FMT<0b000, 0b001001>;
class SLLI_W_ENC : MSA_BIT_W_FMT<0b000, 0b001001>;
class SLLI_D_ENC : MSA_BIT_D_FMT<0b000, 0b001001>;

class SPLAT_B_ENC : MSA_3R_FMT<0b001, 0b00, 0b010100>;
class SPLAT_H_ENC : MSA_3R_FMT<0b001, 0b01, 0b010100>;
class SPLAT_W_ENC : MSA_3R_FMT<0b001, 0b10, 0b010100>;
class SPLAT_D_ENC : MSA_3R_FMT<0b001, 0b11, 0b010100>;

class SPLATI_B_ENC : MSA_ELM_B_FMT<0b0001, 0b011001>;
class SPLATI_H_ENC : MSA_ELM_H_FMT<0b0001, 0b011001>;
class SPLATI_W_ENC : MSA_ELM_W_FMT<0b0001, 0b011001>;
class SPLATI_D_ENC : MSA_ELM_D_FMT<0b0001, 0b011001>;

class SRA_B_ENC : MSA_3R_FMT<0b001, 0b00, 0b001101>;
class SRA_H_ENC : MSA_3R_FMT<0b001, 0b01, 0b001101>;
class SRA_W_ENC : MSA_3R_FMT<0b001, 0b10, 0b001101>;
class SRA_D_ENC : MSA_3R_FMT<0b001, 0b11, 0b001101>;

class SRAI_B_ENC : MSA_BIT_B_FMT<0b001, 0b001001>;
class SRAI_H_ENC : MSA_BIT_H_FMT<0b001, 0b001001>;
class SRAI_W_ENC : MSA_BIT_W_FMT<0b001, 0b001001>;
class SRAI_D_ENC : MSA_BIT_D_FMT<0b001, 0b001001>;

class SRAR_B_ENC : MSA_3R_FMT<0b001, 0b00, 0b010101>;
class SRAR_H_ENC : MSA_3R_FMT<0b001, 0b01, 0b010101>;
class SRAR_W_ENC : MSA_3R_FMT<0b001, 0b10, 0b010101>;
class SRAR_D_ENC : MSA_3R_FMT<0b001, 0b11, 0b010101>;

class SRARI_B_ENC : MSA_BIT_B_FMT<0b010, 0b001010>;
class SRARI_H_ENC : MSA_BIT_H_FMT<0b010, 0b001010>;
class SRARI_W_ENC : MSA_BIT_W_FMT<0b010, 0b001010>;
class SRARI_D_ENC : MSA_BIT_D_FMT<0b010, 0b001010>;

class SRL_B_ENC : MSA_3R_FMT<0b010, 0b00, 0b001101>;
class SRL_H_ENC : MSA_3R_FMT<0b010, 0b01, 0b001101>;
class SRL_W_ENC : MSA_3R_FMT<0b010, 0b10, 0b001101>;
class SRL_D_ENC : MSA_3R_FMT<0b010, 0b11, 0b001101>;

class SRLI_B_ENC : MSA_BIT_B_FMT<0b010, 0b001001>;
class SRLI_H_ENC : MSA_BIT_H_FMT<0b010, 0b001001>;
class SRLI_W_ENC : MSA_BIT_W_FMT<0b010, 0b001001>;
class SRLI_D_ENC : MSA_BIT_D_FMT<0b010, 0b001001>;

class SRLR_B_ENC : MSA_3R_FMT<0b010, 0b00, 0b010101>;
class SRLR_H_ENC : MSA_3R_FMT<0b010, 0b01, 0b010101>;
class SRLR_W_ENC : MSA_3R_FMT<0b010, 0b10, 0b010101>;
class SRLR_D_ENC : MSA_3R_FMT<0b010, 0b11, 0b010101>;

class SRLRI_B_ENC : MSA_BIT_B_FMT<0b011, 0b001010>;
class SRLRI_H_ENC : MSA_BIT_H_FMT<0b011, 0b001010>;
class SRLRI_W_ENC : MSA_BIT_W_FMT<0b011, 0b001010>;
class SRLRI_D_ENC : MSA_BIT_D_FMT<0b011, 0b001010>;

class ST_B_ENC   : MSA_I5_FMT<0b111, 0b00, 0b000111>;
class ST_H_ENC   : MSA_I5_FMT<0b111, 0b01, 0b000111>;
class ST_W_ENC   : MSA_I5_FMT<0b111, 0b10, 0b000111>;
class ST_D_ENC   : MSA_I5_FMT<0b111, 0b11, 0b000111>;

class STX_B_ENC  : MSA_3R_FMT<0b111, 0b00, 0b001111>;
class STX_H_ENC  : MSA_3R_FMT<0b111, 0b01, 0b001111>;
class STX_W_ENC  : MSA_3R_FMT<0b111, 0b10, 0b001111>;
class STX_D_ENC  : MSA_3R_FMT<0b111, 0b11, 0b001111>;

class SUBS_S_B_ENC : MSA_3R_FMT<0b000, 0b00, 0b010001>;
class SUBS_S_H_ENC : MSA_3R_FMT<0b000, 0b01, 0b010001>;
class SUBS_S_W_ENC : MSA_3R_FMT<0b000, 0b10, 0b010001>;
class SUBS_S_D_ENC : MSA_3R_FMT<0b000, 0b11, 0b010001>;

class SUBS_U_B_ENC : MSA_3R_FMT<0b001, 0b00, 0b010001>;
class SUBS_U_H_ENC : MSA_3R_FMT<0b001, 0b01, 0b010001>;
class SUBS_U_W_ENC : MSA_3R_FMT<0b001, 0b10, 0b010001>;
class SUBS_U_D_ENC : MSA_3R_FMT<0b001, 0b11, 0b010001>;

class SUBSUS_U_B_ENC : MSA_3R_FMT<0b011, 0b00, 0b010001>;
class SUBSUS_U_H_ENC : MSA_3R_FMT<0b011, 0b01, 0b010001>;
class SUBSUS_U_W_ENC : MSA_3R_FMT<0b011, 0b10, 0b010001>;
class SUBSUS_U_D_ENC : MSA_3R_FMT<0b011, 0b11, 0b010001>;

class SUBSUU_S_B_ENC : MSA_3R_FMT<0b010, 0b00, 0b010001>;
class SUBSUU_S_H_ENC : MSA_3R_FMT<0b010, 0b01, 0b010001>;
class SUBSUU_S_W_ENC : MSA_3R_FMT<0b010, 0b10, 0b010001>;
class SUBSUU_S_D_ENC : MSA_3R_FMT<0b010, 0b11, 0b010001>;

class SUBV_B_ENC : MSA_3R_FMT<0b001, 0b00, 0b001110>;
class SUBV_H_ENC : MSA_3R_FMT<0b001, 0b01, 0b001110>;
class SUBV_W_ENC : MSA_3R_FMT<0b001, 0b10, 0b001110>;
class SUBV_D_ENC : MSA_3R_FMT<0b001, 0b11, 0b001110>;

class SUBVI_B_ENC : MSA_I5_FMT<0b001, 0b00, 0b000110>;
class SUBVI_H_ENC : MSA_I5_FMT<0b001, 0b01, 0b000110>;
class SUBVI_W_ENC : MSA_I5_FMT<0b001, 0b10, 0b000110>;
class SUBVI_D_ENC : MSA_I5_FMT<0b001, 0b11, 0b000110>;

class VSHF_B_ENC : MSA_3R_FMT<0b000, 0b00, 0b010101>;
class VSHF_H_ENC : MSA_3R_FMT<0b000, 0b01, 0b010101>;
class VSHF_W_ENC : MSA_3R_FMT<0b000, 0b10, 0b010101>;
class VSHF_D_ENC : MSA_3R_FMT<0b000, 0b11, 0b010101>;

class XOR_V_ENC : MSA_VEC_FMT<0b00011, 0b011110>;

class XORI_B_ENC : MSA_I8_FMT<0b11, 0b000000>;

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

class MSA_CBRANCH_DESC_BASE<string instr_asm, RegisterClass RCWD> {
  dag OutOperandList = (outs);
  dag InOperandList = (ins RCWD:$wd, brtarget:$offset);
  string AsmString = !strconcat(instr_asm, "\t$wd, $offset");
  list<dag> Pattern = [];
  InstrItinClass Itinerary = IIBranch;
  bit isBranch = 1;
  bit isTerminator = 1;
  bit hasDelaySlot = 1;
  list<Register> Defs = [AT];
}

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

class MSA_INSVE_DESC_BASE<string instr_asm, SDPatternOperator OpNode,
                          InstrItinClass itin, RegisterClass RCD,
                          RegisterClass RCWS> {
  dag OutOperandList = (outs RCD:$wd);
  dag InOperandList = (ins RCD:$wd_in, uimm6:$n, RCWS:$ws);
  string AsmString = !strconcat(instr_asm, "\t$wd[$n], $ws[0]");
  list<dag> Pattern = [(set RCD:$wd, (OpNode RCD:$wd_in,
                                             immZExt6:$n,
                                             RCWS:$ws))];
  InstrItinClass Itinerary = itin;
  string Constraints = "$wd = $wd_in";
}

class MSA_VEC_DESC_BASE<string instr_asm, SDPatternOperator OpNode,
                        InstrItinClass itin, RegisterClass RCWD,
                        RegisterClass RCWS,  RegisterClass RCWT = RCWS> {
  dag OutOperandList = (outs RCWD:$wd);
  dag InOperandList = (ins RCWS:$ws, RCWT:$wt);
  string AsmString = !strconcat(instr_asm, "\t$wd, $ws, $wt");
  list<dag> Pattern = [(set RCWD:$wd, (OpNode RCWS:$ws, RCWT:$wt))];
  InstrItinClass Itinerary = itin;
}

class ADD_A_B_DESC : MSA_3R_DESC_BASE<"add_a.b", int_mips_add_a_b, NoItinerary,
                                      MSA128B, MSA128B>, IsCommutable;
class ADD_A_H_DESC : MSA_3R_DESC_BASE<"add_a.h", int_mips_add_a_h, NoItinerary,
                                      MSA128H, MSA128H>, IsCommutable;
class ADD_A_W_DESC : MSA_3R_DESC_BASE<"add_a.w", int_mips_add_a_w, NoItinerary,
                                      MSA128W, MSA128W>, IsCommutable;
class ADD_A_D_DESC : MSA_3R_DESC_BASE<"add_a.d", int_mips_add_a_d, NoItinerary,
                                      MSA128D, MSA128D>, IsCommutable;

class ADDS_A_B_DESC : MSA_3R_DESC_BASE<"adds_a.b", int_mips_adds_a_b,
                                       NoItinerary, MSA128B, MSA128B>,
                                       IsCommutable;
class ADDS_A_H_DESC : MSA_3R_DESC_BASE<"adds_a.h", int_mips_adds_a_h,
                                       NoItinerary, MSA128H, MSA128H>,
                                       IsCommutable;
class ADDS_A_W_DESC : MSA_3R_DESC_BASE<"adds_a.w", int_mips_adds_a_w,
                                       NoItinerary, MSA128W, MSA128W>,
                                       IsCommutable;
class ADDS_A_D_DESC : MSA_3R_DESC_BASE<"adds_a.d", int_mips_adds_a_d,
                                       NoItinerary, MSA128D, MSA128D>,
                                       IsCommutable;

class ADDS_S_B_DESC : MSA_3R_DESC_BASE<"adds_s.b", int_mips_adds_s_b,
                                       NoItinerary, MSA128B, MSA128B>,
                                       IsCommutable;
class ADDS_S_H_DESC : MSA_3R_DESC_BASE<"adds_s.h", int_mips_adds_s_h,
                                       NoItinerary, MSA128H, MSA128H>,
                                       IsCommutable;
class ADDS_S_W_DESC : MSA_3R_DESC_BASE<"adds_s.w", int_mips_adds_s_w,
                                       NoItinerary, MSA128W, MSA128W>,
                                       IsCommutable;
class ADDS_S_D_DESC : MSA_3R_DESC_BASE<"adds_s.d", int_mips_adds_s_d,
                                       NoItinerary, MSA128D, MSA128D>,
                                       IsCommutable;

class ADDS_U_B_DESC : MSA_3R_DESC_BASE<"adds_u.b", int_mips_adds_u_b,
                                       NoItinerary, MSA128B, MSA128B>,
                                       IsCommutable;
class ADDS_U_H_DESC : MSA_3R_DESC_BASE<"adds_u.h", int_mips_adds_u_h,
                                       NoItinerary, MSA128H, MSA128H>,
                                       IsCommutable;
class ADDS_U_W_DESC : MSA_3R_DESC_BASE<"adds_u.w", int_mips_adds_u_w,
                                       NoItinerary, MSA128W, MSA128W>,
                                       IsCommutable;
class ADDS_U_D_DESC : MSA_3R_DESC_BASE<"adds_u.d", int_mips_adds_u_d,
                                       NoItinerary, MSA128D, MSA128D>,
                                       IsCommutable;

class ADDV_B_DESC : MSA_3R_DESC_BASE<"addv.b", int_mips_addv_b, NoItinerary,
                                     MSA128B, MSA128B>, IsCommutable;
class ADDV_H_DESC : MSA_3R_DESC_BASE<"addv.h", int_mips_addv_h, NoItinerary,
                                     MSA128H, MSA128H>, IsCommutable;
class ADDV_W_DESC : MSA_3R_DESC_BASE<"addv.w", int_mips_addv_w, NoItinerary,
                                     MSA128W, MSA128W>, IsCommutable;
class ADDV_D_DESC : MSA_3R_DESC_BASE<"addv.d", int_mips_addv_d, NoItinerary,
                                     MSA128D, MSA128D>, IsCommutable;

class ADDVI_B_DESC : MSA_I5_DESC_BASE<"addvi.b", int_mips_addvi_b, NoItinerary,
                                      MSA128B, MSA128B>;
class ADDVI_H_DESC : MSA_I5_DESC_BASE<"addvi.h", int_mips_addvi_h, NoItinerary,
                                      MSA128H, MSA128H>;
class ADDVI_W_DESC : MSA_I5_DESC_BASE<"addvi.w", int_mips_addvi_w, NoItinerary,
                                      MSA128W, MSA128W>;
class ADDVI_D_DESC : MSA_I5_DESC_BASE<"addvi.d", int_mips_addvi_d, NoItinerary,
                                      MSA128D, MSA128D>;

class AND_V_DESC : MSA_VEC_DESC_BASE<"and.v", int_mips_and_v, NoItinerary,
                                     MSA128B, MSA128B>;

class ANDI_B_DESC : MSA_I8_DESC_BASE<"andi.b", int_mips_andi_b, NoItinerary,
                                     MSA128B, MSA128B>;

class ASUB_S_B_DESC : MSA_3R_DESC_BASE<"asub_s.b", int_mips_asub_s_b,
                                       NoItinerary, MSA128B, MSA128B>;
class ASUB_S_H_DESC : MSA_3R_DESC_BASE<"asub_s.h", int_mips_asub_s_h,
                                       NoItinerary, MSA128H, MSA128H>;
class ASUB_S_W_DESC : MSA_3R_DESC_BASE<"asub_s.w", int_mips_asub_s_w,
                                       NoItinerary, MSA128W, MSA128W>;
class ASUB_S_D_DESC : MSA_3R_DESC_BASE<"asub_s.d", int_mips_asub_s_d,
                                       NoItinerary, MSA128D, MSA128D>;

class ASUB_U_B_DESC : MSA_3R_DESC_BASE<"asub_u.b", int_mips_asub_u_b,
                                       NoItinerary, MSA128B, MSA128B>;
class ASUB_U_H_DESC : MSA_3R_DESC_BASE<"asub_u.h", int_mips_asub_u_h,
                                       NoItinerary, MSA128H, MSA128H>;
class ASUB_U_W_DESC : MSA_3R_DESC_BASE<"asub_u.w", int_mips_asub_u_w,
                                       NoItinerary, MSA128W, MSA128W>;
class ASUB_U_D_DESC : MSA_3R_DESC_BASE<"asub_u.d", int_mips_asub_u_d,
                                       NoItinerary, MSA128D, MSA128D>;

class AVE_S_B_DESC : MSA_3R_DESC_BASE<"ave_s.b", int_mips_ave_s_b, NoItinerary,
                                       MSA128B, MSA128B>, IsCommutable;
class AVE_S_H_DESC : MSA_3R_DESC_BASE<"ave_s.h", int_mips_ave_s_h, NoItinerary,
                                       MSA128H, MSA128H>, IsCommutable;
class AVE_S_W_DESC : MSA_3R_DESC_BASE<"ave_s.w", int_mips_ave_s_w, NoItinerary,
                                       MSA128W, MSA128W>, IsCommutable;
class AVE_S_D_DESC : MSA_3R_DESC_BASE<"ave_s.d", int_mips_ave_s_d, NoItinerary,
                                       MSA128D, MSA128D>, IsCommutable;

class AVE_U_B_DESC : MSA_3R_DESC_BASE<"ave_u.b", int_mips_ave_u_b, NoItinerary,
                                       MSA128B, MSA128B>, IsCommutable;
class AVE_U_H_DESC : MSA_3R_DESC_BASE<"ave_u.h", int_mips_ave_u_h, NoItinerary,
                                       MSA128H, MSA128H>, IsCommutable;
class AVE_U_W_DESC : MSA_3R_DESC_BASE<"ave_u.w", int_mips_ave_u_w, NoItinerary,
                                       MSA128W, MSA128W>, IsCommutable;
class AVE_U_D_DESC : MSA_3R_DESC_BASE<"ave_u.d", int_mips_ave_u_d, NoItinerary,
                                       MSA128D, MSA128D>, IsCommutable;

class AVER_S_B_DESC : MSA_3R_DESC_BASE<"aver_s.b", int_mips_aver_s_b,
                                       NoItinerary, MSA128B, MSA128B>,
                                       IsCommutable;
class AVER_S_H_DESC : MSA_3R_DESC_BASE<"aver_s.h", int_mips_aver_s_h,
                                       NoItinerary, MSA128H, MSA128H>,
                                       IsCommutable;
class AVER_S_W_DESC : MSA_3R_DESC_BASE<"aver_s.w", int_mips_aver_s_w,
                                       NoItinerary, MSA128W, MSA128W>,
                                       IsCommutable;
class AVER_S_D_DESC : MSA_3R_DESC_BASE<"aver_s.d", int_mips_aver_s_d,
                                       NoItinerary, MSA128D, MSA128D>,
                                       IsCommutable;

class AVER_U_B_DESC : MSA_3R_DESC_BASE<"aver_u.b", int_mips_aver_u_b,
                                       NoItinerary, MSA128B, MSA128B>,
                                       IsCommutable;
class AVER_U_H_DESC : MSA_3R_DESC_BASE<"aver_u.h", int_mips_aver_u_h,
                                       NoItinerary, MSA128H, MSA128H>,
                                       IsCommutable;
class AVER_U_W_DESC : MSA_3R_DESC_BASE<"aver_u.w", int_mips_aver_u_w,
                                       NoItinerary, MSA128W, MSA128W>,
                                       IsCommutable;
class AVER_U_D_DESC : MSA_3R_DESC_BASE<"aver_u.d", int_mips_aver_u_d,
                                       NoItinerary, MSA128D, MSA128D>,
                                       IsCommutable;

class BCLR_B_DESC : MSA_3R_DESC_BASE<"bclr.b", int_mips_bclr_b, NoItinerary,
                                     MSA128B, MSA128B>;
class BCLR_H_DESC : MSA_3R_DESC_BASE<"bclr.h", int_mips_bclr_h, NoItinerary,
                                     MSA128H, MSA128H>;
class BCLR_W_DESC : MSA_3R_DESC_BASE<"bclr.w", int_mips_bclr_w, NoItinerary,
                                     MSA128W, MSA128W>;
class BCLR_D_DESC : MSA_3R_DESC_BASE<"bclr.d", int_mips_bclr_d, NoItinerary,
                                     MSA128D, MSA128D>;

class BCLRI_B_DESC : MSA_BIT_B_DESC_BASE<"bclri.b", int_mips_bclri_b,
                                         NoItinerary, MSA128B, MSA128B>;
class BCLRI_H_DESC : MSA_BIT_H_DESC_BASE<"bclri.h", int_mips_bclri_h,
                                         NoItinerary, MSA128H, MSA128H>;
class BCLRI_W_DESC : MSA_BIT_W_DESC_BASE<"bclri.w", int_mips_bclri_w,
                                         NoItinerary, MSA128W, MSA128W>;
class BCLRI_D_DESC : MSA_BIT_D_DESC_BASE<"bclri.d", int_mips_bclri_d,
                                         NoItinerary, MSA128D, MSA128D>;

class BINSL_B_DESC : MSA_3R_DESC_BASE<"binsl.b", int_mips_binsl_b, NoItinerary,
                                      MSA128B, MSA128B>;
class BINSL_H_DESC : MSA_3R_DESC_BASE<"binsl.h", int_mips_binsl_h, NoItinerary,
                                      MSA128H, MSA128H>;
class BINSL_W_DESC : MSA_3R_DESC_BASE<"binsl.w", int_mips_binsl_w, NoItinerary,
                                      MSA128W, MSA128W>;
class BINSL_D_DESC : MSA_3R_DESC_BASE<"binsl.d", int_mips_binsl_d, NoItinerary,
                                      MSA128D, MSA128D>;

class BINSLI_B_DESC : MSA_BIT_B_DESC_BASE<"binsli.b", int_mips_binsli_b,
                                          NoItinerary, MSA128B, MSA128B>;
class BINSLI_H_DESC : MSA_BIT_H_DESC_BASE<"binsli.h", int_mips_binsli_h,
                                          NoItinerary, MSA128H, MSA128H>;
class BINSLI_W_DESC : MSA_BIT_W_DESC_BASE<"binsli.w", int_mips_binsli_w,
                                          NoItinerary, MSA128W, MSA128W>;
class BINSLI_D_DESC : MSA_BIT_D_DESC_BASE<"binsli.d", int_mips_binsli_d,
                                          NoItinerary, MSA128D, MSA128D>;

class BINSR_B_DESC : MSA_3R_DESC_BASE<"binsr.b", int_mips_binsr_b, NoItinerary,
                                      MSA128B, MSA128B>;
class BINSR_H_DESC : MSA_3R_DESC_BASE<"binsr.h", int_mips_binsr_h, NoItinerary,
                                      MSA128H, MSA128H>;
class BINSR_W_DESC : MSA_3R_DESC_BASE<"binsr.w", int_mips_binsr_w, NoItinerary,
                                      MSA128W, MSA128W>;
class BINSR_D_DESC : MSA_3R_DESC_BASE<"binsr.d", int_mips_binsr_d, NoItinerary,
                                      MSA128D, MSA128D>;

class BINSRI_B_DESC : MSA_BIT_B_DESC_BASE<"binsri.b", int_mips_binsri_b,
                                          NoItinerary, MSA128B, MSA128B>;
class BINSRI_H_DESC : MSA_BIT_H_DESC_BASE<"binsri.h", int_mips_binsri_h,
                                          NoItinerary, MSA128H, MSA128H>;
class BINSRI_W_DESC : MSA_BIT_W_DESC_BASE<"binsri.w", int_mips_binsri_w,
                                          NoItinerary, MSA128W, MSA128W>;
class BINSRI_D_DESC : MSA_BIT_D_DESC_BASE<"binsri.d", int_mips_binsri_d,
                                          NoItinerary, MSA128D, MSA128D>;

class BMNZ_V_DESC : MSA_VEC_DESC_BASE<"bmnz.v", int_mips_bmnz_v, NoItinerary,
                                      MSA128B, MSA128B>;

class BMNZI_B_DESC : MSA_I8_DESC_BASE<"bmnzi.b", int_mips_bmnzi_b, NoItinerary,
                                      MSA128B, MSA128B>;

class BMZ_V_DESC : MSA_VEC_DESC_BASE<"bmz.v", int_mips_bmz_v, NoItinerary,
                                     MSA128B, MSA128B>;

class BMZI_B_DESC : MSA_I8_DESC_BASE<"bmzi.b", int_mips_bmzi_b, NoItinerary,
                                     MSA128B, MSA128B>;

class BNEG_B_DESC : MSA_3R_DESC_BASE<"bneg.b", int_mips_bneg_b, NoItinerary,
                                     MSA128B, MSA128B>;
class BNEG_H_DESC : MSA_3R_DESC_BASE<"bneg.h", int_mips_bneg_h, NoItinerary,
                                     MSA128H, MSA128H>;
class BNEG_W_DESC : MSA_3R_DESC_BASE<"bneg.w", int_mips_bneg_w, NoItinerary,
                                     MSA128W, MSA128W>;
class BNEG_D_DESC : MSA_3R_DESC_BASE<"bneg.d", int_mips_bneg_d, NoItinerary,
                                     MSA128D, MSA128D>;

class BNEGI_B_DESC : MSA_BIT_B_DESC_BASE<"bnegi.b", int_mips_bnegi_b,
                                         NoItinerary, MSA128B, MSA128B>;
class BNEGI_H_DESC : MSA_BIT_H_DESC_BASE<"bnegi.h", int_mips_bnegi_h,
                                         NoItinerary, MSA128H, MSA128H>;
class BNEGI_W_DESC : MSA_BIT_W_DESC_BASE<"bnegi.w", int_mips_bnegi_w,
                                         NoItinerary, MSA128W, MSA128W>;
class BNEGI_D_DESC : MSA_BIT_D_DESC_BASE<"bnegi.d", int_mips_bnegi_d,
                                         NoItinerary, MSA128D, MSA128D>;

class BNZ_B_DESC : MSA_CBRANCH_DESC_BASE<"bnz.b", MSA128B>;
class BNZ_H_DESC : MSA_CBRANCH_DESC_BASE<"bnz.h", MSA128H>;
class BNZ_W_DESC : MSA_CBRANCH_DESC_BASE<"bnz.w", MSA128W>;
class BNZ_D_DESC : MSA_CBRANCH_DESC_BASE<"bnz.d", MSA128D>;

class BNZ_V_DESC : MSA_CBRANCH_DESC_BASE<"bnz.v", MSA128B>;

class BSEL_V_DESC : MSA_VEC_DESC_BASE<"bsel.v", int_mips_bsel_v, NoItinerary,
                                      MSA128B, MSA128B>;

class BSELI_B_DESC : MSA_I8_DESC_BASE<"bseli.b", int_mips_bseli_b, NoItinerary,
                                      MSA128B, MSA128B>;

class BSET_B_DESC : MSA_3R_DESC_BASE<"bset.b", int_mips_bset_b, NoItinerary,
                                     MSA128B, MSA128B>;
class BSET_H_DESC : MSA_3R_DESC_BASE<"bset.h", int_mips_bset_h, NoItinerary,
                                     MSA128H, MSA128H>;
class BSET_W_DESC : MSA_3R_DESC_BASE<"bset.w", int_mips_bset_w, NoItinerary,
                                     MSA128W, MSA128W>;
class BSET_D_DESC : MSA_3R_DESC_BASE<"bset.d", int_mips_bset_d, NoItinerary,
                                     MSA128D, MSA128D>;

class BSETI_B_DESC : MSA_BIT_B_DESC_BASE<"bseti.b", int_mips_bseti_b,
                                         NoItinerary, MSA128B, MSA128B>;
class BSETI_H_DESC : MSA_BIT_H_DESC_BASE<"bseti.h", int_mips_bseti_h,
                                         NoItinerary, MSA128H, MSA128H>;
class BSETI_W_DESC : MSA_BIT_W_DESC_BASE<"bseti.w", int_mips_bseti_w,
                                         NoItinerary, MSA128W, MSA128W>;
class BSETI_D_DESC : MSA_BIT_D_DESC_BASE<"bseti.d", int_mips_bseti_d,
                                         NoItinerary, MSA128D, MSA128D>;

class BZ_B_DESC : MSA_CBRANCH_DESC_BASE<"bz.b", MSA128B>;
class BZ_H_DESC : MSA_CBRANCH_DESC_BASE<"bz.h", MSA128H>;
class BZ_W_DESC : MSA_CBRANCH_DESC_BASE<"bz.w", MSA128W>;
class BZ_D_DESC : MSA_CBRANCH_DESC_BASE<"bz.d", MSA128D>;

class BZ_V_DESC : MSA_CBRANCH_DESC_BASE<"bz.v", MSA128B>;

class CEQ_B_DESC : MSA_3R_DESC_BASE<"ceq.b", int_mips_ceq_b, NoItinerary,
                                    MSA128B, MSA128B>, IsCommutable;
class CEQ_H_DESC : MSA_3R_DESC_BASE<"ceq.h", int_mips_ceq_h, NoItinerary,
                                    MSA128H, MSA128H>, IsCommutable;
class CEQ_W_DESC : MSA_3R_DESC_BASE<"ceq.w", int_mips_ceq_w, NoItinerary,
                                    MSA128W, MSA128W>, IsCommutable;
class CEQ_D_DESC : MSA_3R_DESC_BASE<"ceq.d", int_mips_ceq_d, NoItinerary,
                                    MSA128D, MSA128D>, IsCommutable;

class CEQI_B_DESC : MSA_SI5_DESC_BASE<"ceqi.b", int_mips_ceqi_b, NoItinerary,
                                      MSA128B, MSA128B>;
class CEQI_H_DESC : MSA_SI5_DESC_BASE<"ceqi.h", int_mips_ceqi_h, NoItinerary,
                                      MSA128H, MSA128H>;
class CEQI_W_DESC : MSA_SI5_DESC_BASE<"ceqi.w", int_mips_ceqi_w, NoItinerary,
                                      MSA128W, MSA128W>;
class CEQI_D_DESC : MSA_SI5_DESC_BASE<"ceqi.d", int_mips_ceqi_d, NoItinerary,
                                      MSA128D, MSA128D>;

class CFCMSA_DESC {
  dag OutOperandList = (outs GPR32:$rd);
  dag InOperandList = (ins MSACtrl:$cs);
  string AsmString = "cfcmsa\t$rd, $cs";
  InstrItinClass Itinerary = NoItinerary;
  bit hasSideEffects = 1;
}

class CLE_S_B_DESC : MSA_3R_DESC_BASE<"cle_s.b", int_mips_cle_s_b, NoItinerary,
                                      MSA128B, MSA128B>;
class CLE_S_H_DESC : MSA_3R_DESC_BASE<"cle_s.h", int_mips_cle_s_h, NoItinerary,
                                      MSA128H, MSA128H>;
class CLE_S_W_DESC : MSA_3R_DESC_BASE<"cle_s.w", int_mips_cle_s_w, NoItinerary,
                                      MSA128W, MSA128W>;
class CLE_S_D_DESC : MSA_3R_DESC_BASE<"cle_s.d", int_mips_cle_s_d, NoItinerary,
                                      MSA128D, MSA128D>;

class CLE_U_B_DESC : MSA_3R_DESC_BASE<"cle_u.b", int_mips_cle_u_b, NoItinerary,
                                      MSA128B, MSA128B>;
class CLE_U_H_DESC : MSA_3R_DESC_BASE<"cle_u.h", int_mips_cle_u_h, NoItinerary,
                                      MSA128H, MSA128H>;
class CLE_U_W_DESC : MSA_3R_DESC_BASE<"cle_u.w", int_mips_cle_u_w, NoItinerary,
                                      MSA128W, MSA128W>;
class CLE_U_D_DESC : MSA_3R_DESC_BASE<"cle_u.d", int_mips_cle_u_d, NoItinerary,
                                      MSA128D, MSA128D>;

class CLEI_S_B_DESC : MSA_SI5_DESC_BASE<"clei_s.b", int_mips_clei_s_b,
                                        NoItinerary, MSA128B, MSA128B>;
class CLEI_S_H_DESC : MSA_SI5_DESC_BASE<"clei_s.h", int_mips_clei_s_h,
                                        NoItinerary, MSA128H, MSA128H>;
class CLEI_S_W_DESC : MSA_SI5_DESC_BASE<"clei_s.w", int_mips_clei_s_w,
                                        NoItinerary, MSA128W, MSA128W>;
class CLEI_S_D_DESC : MSA_SI5_DESC_BASE<"clei_s.d", int_mips_clei_s_d,
                                        NoItinerary, MSA128D, MSA128D>;

class CLEI_U_B_DESC : MSA_SI5_DESC_BASE<"clei_u.b", int_mips_clei_u_b,
                                        NoItinerary, MSA128B, MSA128B>;
class CLEI_U_H_DESC : MSA_SI5_DESC_BASE<"clei_u.h", int_mips_clei_u_h,
                                        NoItinerary, MSA128H, MSA128H>;
class CLEI_U_W_DESC : MSA_SI5_DESC_BASE<"clei_u.w", int_mips_clei_u_w,
                                        NoItinerary, MSA128W, MSA128W>;
class CLEI_U_D_DESC : MSA_SI5_DESC_BASE<"clei_u.d", int_mips_clei_u_d,
                                        NoItinerary, MSA128D, MSA128D>;

class CLT_S_B_DESC : MSA_3R_DESC_BASE<"clt_s.b", int_mips_clt_s_b, NoItinerary,
                                      MSA128B, MSA128B>;
class CLT_S_H_DESC : MSA_3R_DESC_BASE<"clt_s.h", int_mips_clt_s_h, NoItinerary,
                                      MSA128H, MSA128H>;
class CLT_S_W_DESC : MSA_3R_DESC_BASE<"clt_s.w", int_mips_clt_s_w, NoItinerary,
                                      MSA128W, MSA128W>;
class CLT_S_D_DESC : MSA_3R_DESC_BASE<"clt_s.d", int_mips_clt_s_d, NoItinerary,
                                      MSA128D, MSA128D>;

class CLT_U_B_DESC : MSA_3R_DESC_BASE<"clt_u.b", int_mips_clt_u_b, NoItinerary,
                                      MSA128B, MSA128B>;
class CLT_U_H_DESC : MSA_3R_DESC_BASE<"clt_u.h", int_mips_clt_u_h, NoItinerary,
                                      MSA128H, MSA128H>;
class CLT_U_W_DESC : MSA_3R_DESC_BASE<"clt_u.w", int_mips_clt_u_w, NoItinerary,
                                      MSA128W, MSA128W>;
class CLT_U_D_DESC : MSA_3R_DESC_BASE<"clt_u.d", int_mips_clt_u_d, NoItinerary,
                                      MSA128D, MSA128D>;

class CLTI_S_B_DESC : MSA_SI5_DESC_BASE<"clti_s.b", int_mips_clti_s_b,
                                        NoItinerary, MSA128B, MSA128B>;
class CLTI_S_H_DESC : MSA_SI5_DESC_BASE<"clti_s.h", int_mips_clti_s_h,
                                        NoItinerary, MSA128H, MSA128H>;
class CLTI_S_W_DESC : MSA_SI5_DESC_BASE<"clti_s.w", int_mips_clti_s_w,
                                        NoItinerary, MSA128W, MSA128W>;
class CLTI_S_D_DESC : MSA_SI5_DESC_BASE<"clti_s.d", int_mips_clti_s_d,
                                        NoItinerary, MSA128D, MSA128D>;

class CLTI_U_B_DESC : MSA_SI5_DESC_BASE<"clti_u.b", int_mips_clti_u_b,
                                        NoItinerary, MSA128B, MSA128B>;
class CLTI_U_H_DESC : MSA_SI5_DESC_BASE<"clti_u.h", int_mips_clti_u_h,
                                        NoItinerary, MSA128H, MSA128H>;
class CLTI_U_W_DESC : MSA_SI5_DESC_BASE<"clti_u.w", int_mips_clti_u_w,
                                        NoItinerary, MSA128W, MSA128W>;
class CLTI_U_D_DESC : MSA_SI5_DESC_BASE<"clti_u.d", int_mips_clti_u_d,
                                        NoItinerary, MSA128D, MSA128D>;

class COPY_S_B_DESC : MSA_COPY_DESC_BASE<"copy_s.b", int_mips_copy_s_b,
                                         NoItinerary, GPR32, MSA128B>;
class COPY_S_H_DESC : MSA_COPY_DESC_BASE<"copy_s.h", int_mips_copy_s_h,
                                         NoItinerary, GPR32, MSA128H>;
class COPY_S_W_DESC : MSA_COPY_DESC_BASE<"copy_s.w", int_mips_copy_s_w,
                                         NoItinerary, GPR32, MSA128W>;

class COPY_U_B_DESC : MSA_COPY_DESC_BASE<"copy_u.b", int_mips_copy_u_b,
                                         NoItinerary, GPR32, MSA128B>;
class COPY_U_H_DESC : MSA_COPY_DESC_BASE<"copy_u.h", int_mips_copy_u_h,
                                         NoItinerary, GPR32, MSA128H>;
class COPY_U_W_DESC : MSA_COPY_DESC_BASE<"copy_u.w", int_mips_copy_u_w,
                                         NoItinerary, GPR32, MSA128W>;

class CTCMSA_DESC {
  dag OutOperandList = (outs);
  dag InOperandList = (ins MSACtrl:$cd, GPR32:$rs);
  string AsmString = "ctcmsa\t$cd, $rs";
  InstrItinClass Itinerary = NoItinerary;
  bit hasSideEffects = 1;
}

class DIV_S_B_DESC : MSA_3R_DESC_BASE<"div_s.b", int_mips_div_s_b, NoItinerary,
                                      MSA128B, MSA128B>;
class DIV_S_H_DESC : MSA_3R_DESC_BASE<"div_s.h", int_mips_div_s_h, NoItinerary,
                                      MSA128H, MSA128H>;
class DIV_S_W_DESC : MSA_3R_DESC_BASE<"div_s.w", int_mips_div_s_w, NoItinerary,
                                      MSA128W, MSA128W>;
class DIV_S_D_DESC : MSA_3R_DESC_BASE<"div_s.d", int_mips_div_s_d, NoItinerary,
                                      MSA128D, MSA128D>;

class DIV_U_B_DESC : MSA_3R_DESC_BASE<"div_u.b", int_mips_div_u_b, NoItinerary,
                                      MSA128B, MSA128B>;
class DIV_U_H_DESC : MSA_3R_DESC_BASE<"div_u.h", int_mips_div_u_h, NoItinerary,
                                      MSA128H, MSA128H>;
class DIV_U_W_DESC : MSA_3R_DESC_BASE<"div_u.w", int_mips_div_u_w, NoItinerary,
                                      MSA128W, MSA128W>;
class DIV_U_D_DESC : MSA_3R_DESC_BASE<"div_u.d", int_mips_div_u_d, NoItinerary,
                                      MSA128D, MSA128D>;

class DOTP_S_B_DESC : MSA_3R_DESC_BASE<"dotp_s.b", int_mips_dotp_s_b,
                                       NoItinerary, MSA128B, MSA128B>,
                                       IsCommutable;
class DOTP_S_H_DESC : MSA_3R_DESC_BASE<"dotp_s.h", int_mips_dotp_s_h,
                                       NoItinerary, MSA128H, MSA128H>,
                                       IsCommutable;
class DOTP_S_W_DESC : MSA_3R_DESC_BASE<"dotp_s.w", int_mips_dotp_s_w,
                                       NoItinerary, MSA128W, MSA128W>,
                                       IsCommutable;
class DOTP_S_D_DESC : MSA_3R_DESC_BASE<"dotp_s.d", int_mips_dotp_s_d,
                                       NoItinerary, MSA128D, MSA128D>,
                                       IsCommutable;

class DOTP_U_B_DESC : MSA_3R_DESC_BASE<"dotp_u.b", int_mips_dotp_u_b,
                                       NoItinerary, MSA128B, MSA128B>,
                                       IsCommutable;
class DOTP_U_H_DESC : MSA_3R_DESC_BASE<"dotp_u.h", int_mips_dotp_u_h,
                                       NoItinerary, MSA128H, MSA128H>,
                                       IsCommutable;
class DOTP_U_W_DESC : MSA_3R_DESC_BASE<"dotp_u.w", int_mips_dotp_u_w,
                                       NoItinerary, MSA128W, MSA128W>,
                                       IsCommutable;
class DOTP_U_D_DESC : MSA_3R_DESC_BASE<"dotp_u.d", int_mips_dotp_u_d,
                                       NoItinerary, MSA128D, MSA128D>,
                                       IsCommutable;

class DPADD_S_H_DESC : MSA_3R_4R_DESC_BASE<"dpadd_s.h", int_mips_dpadd_s_h,
                                           NoItinerary, MSA128H, MSA128B>,
                                           IsCommutable;
class DPADD_S_W_DESC : MSA_3R_4R_DESC_BASE<"dpadd_s.w", int_mips_dpadd_s_w,
                                           NoItinerary, MSA128W, MSA128H>,
                                           IsCommutable;
class DPADD_S_D_DESC : MSA_3R_4R_DESC_BASE<"dpadd_s.d", int_mips_dpadd_s_d,
                                           NoItinerary, MSA128D, MSA128W>,
                                           IsCommutable;

class DPADD_U_H_DESC : MSA_3R_4R_DESC_BASE<"dpadd_u.h", int_mips_dpadd_u_h,
                                           NoItinerary, MSA128H, MSA128B>,
                                           IsCommutable;
class DPADD_U_W_DESC : MSA_3R_4R_DESC_BASE<"dpadd_u.w", int_mips_dpadd_u_w,
                                           NoItinerary, MSA128W, MSA128H>,
                                           IsCommutable;
class DPADD_U_D_DESC : MSA_3R_4R_DESC_BASE<"dpadd_u.d", int_mips_dpadd_u_d,
                                           NoItinerary, MSA128D, MSA128W>,
                                           IsCommutable;

class DPSUB_S_H_DESC : MSA_3R_4R_DESC_BASE<"dpsub_s.h", int_mips_dpsub_s_h,
                                           NoItinerary, MSA128H, MSA128B>;
class DPSUB_S_W_DESC : MSA_3R_4R_DESC_BASE<"dpsub_s.w", int_mips_dpsub_s_w,
                                           NoItinerary, MSA128W, MSA128H>;
class DPSUB_S_D_DESC : MSA_3R_4R_DESC_BASE<"dpsub_s.d", int_mips_dpsub_s_d,
                                           NoItinerary, MSA128D, MSA128W>;

class DPSUB_U_H_DESC : MSA_3R_4R_DESC_BASE<"dpsub_u.h", int_mips_dpsub_u_h,
                                           NoItinerary, MSA128H, MSA128B>;
class DPSUB_U_W_DESC : MSA_3R_4R_DESC_BASE<"dpsub_u.w", int_mips_dpsub_u_w,
                                           NoItinerary, MSA128W, MSA128H>;
class DPSUB_U_D_DESC : MSA_3R_4R_DESC_BASE<"dpsub_u.d", int_mips_dpsub_u_d,
                                           NoItinerary, MSA128D, MSA128W>;

class FADD_W_DESC : MSA_3RF_DESC_BASE<"fadd.w", int_mips_fadd_w,
                                      NoItinerary, MSA128W, MSA128W>,
                                      IsCommutable;
class FADD_D_DESC : MSA_3RF_DESC_BASE<"fadd.d", int_mips_fadd_d,
                                      NoItinerary, MSA128D, MSA128D>,
                                      IsCommutable;

class FCAF_W_DESC : MSA_3RF_DESC_BASE<"fcaf.w", int_mips_fcaf_w,
                                      NoItinerary, MSA128W, MSA128W>,
                                      IsCommutable;
class FCAF_D_DESC : MSA_3RF_DESC_BASE<"fcaf.d", int_mips_fcaf_d,
                                      NoItinerary, MSA128D, MSA128D>,
                                      IsCommutable;

class FCEQ_W_DESC : MSA_3RF_DESC_BASE<"fceq.w", int_mips_fceq_w,
                                      NoItinerary, MSA128W, MSA128W>,
                                      IsCommutable;
class FCEQ_D_DESC : MSA_3RF_DESC_BASE<"fceq.d", int_mips_fceq_d,
                                      NoItinerary, MSA128D, MSA128D>,
                                      IsCommutable;

class FCLASS_W_DESC : MSA_2RF_DESC_BASE<"fclass.w", int_mips_fclass_w,
                                        NoItinerary, MSA128W, MSA128W>;
class FCLASS_D_DESC : MSA_2RF_DESC_BASE<"fclass.d", int_mips_fclass_d,
                                        NoItinerary, MSA128D, MSA128D>;

class FCLE_W_DESC : MSA_3RF_DESC_BASE<"fcle.w", int_mips_fcle_w,
                                      NoItinerary, MSA128W, MSA128W>;
class FCLE_D_DESC : MSA_3RF_DESC_BASE<"fcle.d", int_mips_fcle_d,
                                      NoItinerary, MSA128D, MSA128D>;

class FCLT_W_DESC : MSA_3RF_DESC_BASE<"fclt.w", int_mips_fclt_w,
                                      NoItinerary, MSA128W, MSA128W>;
class FCLT_D_DESC : MSA_3RF_DESC_BASE<"fclt.d", int_mips_fclt_d,
                                      NoItinerary, MSA128D, MSA128D>;

class FCNE_W_DESC : MSA_3RF_DESC_BASE<"fcne.w", int_mips_fcne_w,
                                      NoItinerary, MSA128W, MSA128W>,
                                      IsCommutable;
class FCNE_D_DESC : MSA_3RF_DESC_BASE<"fcne.d", int_mips_fcne_d,
                                      NoItinerary, MSA128D, MSA128D>,
                                      IsCommutable;

class FCOR_W_DESC : MSA_3RF_DESC_BASE<"fcor.w", int_mips_fcor_w,
                                      NoItinerary, MSA128W, MSA128W>,
                                      IsCommutable;
class FCOR_D_DESC : MSA_3RF_DESC_BASE<"fcor.d", int_mips_fcor_d,
                                      NoItinerary, MSA128D, MSA128D>,
                                      IsCommutable;

class FCUEQ_W_DESC : MSA_3RF_DESC_BASE<"fcueq.w", int_mips_fcueq_w,
                                       NoItinerary, MSA128W, MSA128W>,
                                       IsCommutable;
class FCUEQ_D_DESC : MSA_3RF_DESC_BASE<"fcueq.d", int_mips_fcueq_d,
                                       NoItinerary, MSA128D, MSA128D>,
                                       IsCommutable;

class FCULE_W_DESC : MSA_3RF_DESC_BASE<"fcule.w", int_mips_fcule_w,
                                       NoItinerary, MSA128W, MSA128W>,
                                       IsCommutable;
class FCULE_D_DESC : MSA_3RF_DESC_BASE<"fcule.d", int_mips_fcule_d,
                                       NoItinerary, MSA128D, MSA128D>,
                                       IsCommutable;

class FCULT_W_DESC : MSA_3RF_DESC_BASE<"fcult.w", int_mips_fcult_w,
                                       NoItinerary, MSA128W, MSA128W>,
                                       IsCommutable;
class FCULT_D_DESC : MSA_3RF_DESC_BASE<"fcult.d", int_mips_fcult_d,
                                       NoItinerary, MSA128D, MSA128D>,
                                       IsCommutable;

class FCUN_W_DESC : MSA_3RF_DESC_BASE<"fcun.w", int_mips_fcun_w,
                                      NoItinerary, MSA128W, MSA128W>,
                                      IsCommutable;
class FCUN_D_DESC : MSA_3RF_DESC_BASE<"fcun.d", int_mips_fcun_d,
                                      NoItinerary, MSA128D, MSA128D>,
                                      IsCommutable;

class FCUNE_W_DESC : MSA_3RF_DESC_BASE<"fcune.w", int_mips_fcune_w,
                                       NoItinerary, MSA128W, MSA128W>,
                                       IsCommutable;
class FCUNE_D_DESC : MSA_3RF_DESC_BASE<"fcune.d", int_mips_fcune_d,
                                       NoItinerary, MSA128D, MSA128D>,
                                       IsCommutable;

class FDIV_W_DESC : MSA_3RF_DESC_BASE<"fdiv.w", int_mips_fdiv_w,
                                      NoItinerary, MSA128W, MSA128W>;
class FDIV_D_DESC : MSA_3RF_DESC_BASE<"fdiv.d", int_mips_fdiv_d,
                                      NoItinerary, MSA128D, MSA128D>;

class FEXDO_H_DESC : MSA_3RF_DESC_BASE<"fexdo.h", int_mips_fexdo_h,
                                       NoItinerary, MSA128H, MSA128W>;
class FEXDO_W_DESC : MSA_3RF_DESC_BASE<"fexdo.w", int_mips_fexdo_w,
                                       NoItinerary, MSA128W, MSA128D>;

class FEXP2_W_DESC : MSA_3RF_DESC_BASE<"fexp2.w", int_mips_fexp2_w,
                                       NoItinerary, MSA128W, MSA128W>;
class FEXP2_D_DESC : MSA_3RF_DESC_BASE<"fexp2.d", int_mips_fexp2_d,
                                       NoItinerary, MSA128D, MSA128D>;

class FEXUPL_W_DESC : MSA_2RF_DESC_BASE<"fexupl.w", int_mips_fexupl_w,
                                        NoItinerary, MSA128W, MSA128H>;
class FEXUPL_D_DESC : MSA_2RF_DESC_BASE<"fexupl.d", int_mips_fexupl_d,
                                        NoItinerary, MSA128D, MSA128W>;

class FEXUPR_W_DESC : MSA_2RF_DESC_BASE<"fexupr.w", int_mips_fexupr_w,
                                        NoItinerary, MSA128W, MSA128H>;
class FEXUPR_D_DESC : MSA_2RF_DESC_BASE<"fexupr.d", int_mips_fexupr_d,
                                        NoItinerary, MSA128D, MSA128W>;

class FFINT_S_W_DESC : MSA_2RF_DESC_BASE<"ffint_s.w", int_mips_ffint_s_w,
                                         NoItinerary, MSA128W, MSA128W>;
class FFINT_S_D_DESC : MSA_2RF_DESC_BASE<"ffint_s.d", int_mips_ffint_s_d,
                                         NoItinerary, MSA128D, MSA128D>;

class FFINT_U_W_DESC : MSA_2RF_DESC_BASE<"ffint_u.w", int_mips_ffint_u_w,
                                         NoItinerary, MSA128W, MSA128W>;
class FFINT_U_D_DESC : MSA_2RF_DESC_BASE<"ffint_u.d", int_mips_ffint_u_d,
                                         NoItinerary, MSA128D, MSA128D>;

class FFQL_W_DESC : MSA_2RF_DESC_BASE<"ffql.w", int_mips_ffql_w,
                                      NoItinerary, MSA128W, MSA128H>;
class FFQL_D_DESC : MSA_2RF_DESC_BASE<"ffql.d", int_mips_ffql_d,
                                      NoItinerary, MSA128D, MSA128W>;

class FFQR_W_DESC : MSA_2RF_DESC_BASE<"ffqr.w", int_mips_ffqr_w,
                                      NoItinerary, MSA128W, MSA128H>;
class FFQR_D_DESC : MSA_2RF_DESC_BASE<"ffqr.d", int_mips_ffqr_d,
                                      NoItinerary, MSA128D, MSA128W>;

class FILL_B_DESC : MSA_2R_DESC_BASE<"fill.b", int_mips_fill_b,
                                     NoItinerary, MSA128B, GPR32>;
class FILL_H_DESC : MSA_2R_DESC_BASE<"fill.h", int_mips_fill_h,
                                     NoItinerary, MSA128H, GPR32>;
class FILL_W_DESC : MSA_2R_DESC_BASE<"fill.w", int_mips_fill_w,
                                     NoItinerary, MSA128W, GPR32>;

class FLOG2_W_DESC : MSA_2RF_DESC_BASE<"flog2.w", int_mips_flog2_w,
                                       NoItinerary, MSA128W, MSA128W>;
class FLOG2_D_DESC : MSA_2RF_DESC_BASE<"flog2.d", int_mips_flog2_d,
                                       NoItinerary, MSA128D, MSA128D>;

class FMADD_W_DESC : MSA_3RF_4RF_DESC_BASE<"fmadd.w", int_mips_fmadd_w,
                                           NoItinerary, MSA128W, MSA128W>;
class FMADD_D_DESC : MSA_3RF_4RF_DESC_BASE<"fmadd.d", int_mips_fmadd_d,
                                           NoItinerary, MSA128D, MSA128D>;

class FMAX_W_DESC : MSA_3RF_DESC_BASE<"fmax.w", int_mips_fmax_w,
                                      NoItinerary, MSA128W, MSA128W>;
class FMAX_D_DESC : MSA_3RF_DESC_BASE<"fmax.d", int_mips_fmax_d,
                                      NoItinerary, MSA128D, MSA128D>;

class FMAX_A_W_DESC : MSA_3RF_DESC_BASE<"fmax_a.w", int_mips_fmax_a_w,
                                        NoItinerary, MSA128W, MSA128W>;
class FMAX_A_D_DESC : MSA_3RF_DESC_BASE<"fmax_a.d", int_mips_fmax_a_d,
                                        NoItinerary, MSA128D, MSA128D>;

class FMIN_W_DESC : MSA_3RF_DESC_BASE<"fmin.w", int_mips_fmin_w,
                                      NoItinerary, MSA128W, MSA128W>;
class FMIN_D_DESC : MSA_3RF_DESC_BASE<"fmin.d", int_mips_fmin_d,
                                      NoItinerary, MSA128D, MSA128D>;

class FMIN_A_W_DESC : MSA_3RF_DESC_BASE<"fmin_a.w", int_mips_fmin_a_w,
                                        NoItinerary, MSA128W, MSA128W>;
class FMIN_A_D_DESC : MSA_3RF_DESC_BASE<"fmin_a.d", int_mips_fmin_a_d,
                                        NoItinerary, MSA128D, MSA128D>;

class FMSUB_W_DESC : MSA_3RF_4RF_DESC_BASE<"fmsub.w", int_mips_fmsub_w,
                                           NoItinerary, MSA128W, MSA128W>;
class FMSUB_D_DESC : MSA_3RF_4RF_DESC_BASE<"fmsub.d", int_mips_fmsub_d,
                                           NoItinerary, MSA128D, MSA128D>;

class FMUL_W_DESC : MSA_3RF_DESC_BASE<"fmul.w", int_mips_fmul_w,
                                      NoItinerary, MSA128W, MSA128W>;
class FMUL_D_DESC : MSA_3RF_DESC_BASE<"fmul.d", int_mips_fmul_d,
                                      NoItinerary, MSA128D, MSA128D>;

class FRINT_W_DESC : MSA_2RF_DESC_BASE<"frint.w", int_mips_frint_w,
                                       NoItinerary, MSA128W, MSA128W>;
class FRINT_D_DESC : MSA_2RF_DESC_BASE<"frint.d", int_mips_frint_d,
                                       NoItinerary, MSA128D, MSA128D>;

class FRCP_W_DESC : MSA_2RF_DESC_BASE<"frcp.w", int_mips_frcp_w,
                                       NoItinerary, MSA128W, MSA128W>;
class FRCP_D_DESC : MSA_2RF_DESC_BASE<"frcp.d", int_mips_frcp_d,
                                       NoItinerary, MSA128D, MSA128D>;

class FRSQRT_W_DESC : MSA_2RF_DESC_BASE<"frsqrt.w", int_mips_frsqrt_w,
                                        NoItinerary, MSA128W, MSA128W>;
class FRSQRT_D_DESC : MSA_2RF_DESC_BASE<"frsqrt.d", int_mips_frsqrt_d,
                                        NoItinerary, MSA128D, MSA128D>;

class FSAF_W_DESC : MSA_3RF_DESC_BASE<"fsaf.w", int_mips_fsaf_w,
                                      NoItinerary, MSA128W, MSA128W>;
class FSAF_D_DESC : MSA_3RF_DESC_BASE<"fsaf.d", int_mips_fsaf_d,
                                      NoItinerary, MSA128D, MSA128D>;

class FSEQ_W_DESC : MSA_3RF_DESC_BASE<"fseq.w", int_mips_fseq_w,
                                      NoItinerary, MSA128W, MSA128W>;
class FSEQ_D_DESC : MSA_3RF_DESC_BASE<"fseq.d", int_mips_fseq_d,
                                      NoItinerary, MSA128D, MSA128D>;

class FSLE_W_DESC : MSA_3RF_DESC_BASE<"fsle.w", int_mips_fsle_w,
                                      NoItinerary, MSA128W, MSA128W>;
class FSLE_D_DESC : MSA_3RF_DESC_BASE<"fsle.d", int_mips_fsle_d,
                                      NoItinerary, MSA128D, MSA128D>;

class FSLT_W_DESC : MSA_3RF_DESC_BASE<"fslt.w", int_mips_fslt_w,
                                      NoItinerary, MSA128W, MSA128W>;
class FSLT_D_DESC : MSA_3RF_DESC_BASE<"fslt.d", int_mips_fslt_d,
                                      NoItinerary, MSA128D, MSA128D>;

class FSNE_W_DESC : MSA_3RF_DESC_BASE<"fsne.w", int_mips_fsne_w,
                                      NoItinerary, MSA128W, MSA128W>;
class FSNE_D_DESC : MSA_3RF_DESC_BASE<"fsne.d", int_mips_fsne_d,
                                      NoItinerary, MSA128D, MSA128D>;

class FSOR_W_DESC : MSA_3RF_DESC_BASE<"fsor.w", int_mips_fsor_w,
                                      NoItinerary, MSA128W, MSA128W>;
class FSOR_D_DESC : MSA_3RF_DESC_BASE<"fsor.d", int_mips_fsor_d,
                                      NoItinerary, MSA128D, MSA128D>;

class FSQRT_W_DESC : MSA_2RF_DESC_BASE<"fsqrt.w", int_mips_fsqrt_w,
                                       NoItinerary, MSA128W, MSA128W>;
class FSQRT_D_DESC : MSA_2RF_DESC_BASE<"fsqrt.d", int_mips_fsqrt_d,
                                       NoItinerary, MSA128D, MSA128D>;

class FSUB_W_DESC : MSA_3RF_DESC_BASE<"fsub.w", int_mips_fsub_w,
                                      NoItinerary, MSA128W, MSA128W>;
class FSUB_D_DESC : MSA_3RF_DESC_BASE<"fsub.d", int_mips_fsub_d,
                                      NoItinerary, MSA128D, MSA128D>;

class FSUEQ_W_DESC : MSA_3RF_DESC_BASE<"fsueq.w", int_mips_fsueq_w,
                                       NoItinerary, MSA128W, MSA128W>;
class FSUEQ_D_DESC : MSA_3RF_DESC_BASE<"fsueq.d", int_mips_fsueq_d,
                                       NoItinerary, MSA128D, MSA128D>;

class FSULE_W_DESC : MSA_3RF_DESC_BASE<"fsule.w", int_mips_fsule_w,
                                       NoItinerary, MSA128W, MSA128W>;
class FSULE_D_DESC : MSA_3RF_DESC_BASE<"fsule.d", int_mips_fsule_d,
                                       NoItinerary, MSA128D, MSA128D>;

class FSULT_W_DESC : MSA_3RF_DESC_BASE<"fsult.w", int_mips_fsult_w,
                                       NoItinerary, MSA128W, MSA128W>;
class FSULT_D_DESC : MSA_3RF_DESC_BASE<"fsult.d", int_mips_fsult_d,
                                       NoItinerary, MSA128D, MSA128D>;

class FSUN_W_DESC : MSA_3RF_DESC_BASE<"fsun.w", int_mips_fsun_w,
                                      NoItinerary, MSA128W, MSA128W>;
class FSUN_D_DESC : MSA_3RF_DESC_BASE<"fsun.d", int_mips_fsun_d,
                                      NoItinerary, MSA128D, MSA128D>;

class FSUNE_W_DESC : MSA_3RF_DESC_BASE<"fsune.w", int_mips_fsune_w,
                                       NoItinerary, MSA128W, MSA128W>;
class FSUNE_D_DESC : MSA_3RF_DESC_BASE<"fsune.d", int_mips_fsune_d,
                                       NoItinerary, MSA128D, MSA128D>;

class FTRUNC_S_W_DESC : MSA_2RF_DESC_BASE<"ftrunc_s.w", int_mips_ftrunc_s_w,
                                          NoItinerary, MSA128W, MSA128W>;
class FTRUNC_S_D_DESC : MSA_2RF_DESC_BASE<"ftrunc_s.d", int_mips_ftrunc_s_d,
                                          NoItinerary, MSA128D, MSA128D>;

class FTRUNC_U_W_DESC : MSA_2RF_DESC_BASE<"ftrunc_u.w", int_mips_ftrunc_u_w,
                                          NoItinerary, MSA128W, MSA128W>;
class FTRUNC_U_D_DESC : MSA_2RF_DESC_BASE<"ftrunc_u.d", int_mips_ftrunc_u_d,
                                          NoItinerary, MSA128D, MSA128D>;

class FTINT_S_W_DESC : MSA_2RF_DESC_BASE<"ftint_s.w", int_mips_ftint_s_w,
                                         NoItinerary, MSA128W, MSA128W>;
class FTINT_S_D_DESC : MSA_2RF_DESC_BASE<"ftint_s.d", int_mips_ftint_s_d,
                                         NoItinerary, MSA128D, MSA128D>;

class FTINT_U_W_DESC : MSA_2RF_DESC_BASE<"ftint_u.w", int_mips_ftint_u_w,
                                         NoItinerary, MSA128W, MSA128W>;
class FTINT_U_D_DESC : MSA_2RF_DESC_BASE<"ftint_u.d", int_mips_ftint_u_d,
                                         NoItinerary, MSA128D, MSA128D>;

class FTQ_H_DESC : MSA_3RF_DESC_BASE<"ftq.h", int_mips_ftq_h,
                                     NoItinerary, MSA128H, MSA128W>;
class FTQ_W_DESC : MSA_3RF_DESC_BASE<"ftq.w", int_mips_ftq_w,
                                     NoItinerary, MSA128W, MSA128D>;

class HADD_S_H_DESC : MSA_3R_DESC_BASE<"hadd_s.h", int_mips_hadd_s_h,
                                       NoItinerary, MSA128H, MSA128B>;
class HADD_S_W_DESC : MSA_3R_DESC_BASE<"hadd_s.w", int_mips_hadd_s_w,
                                       NoItinerary, MSA128W, MSA128H>;
class HADD_S_D_DESC : MSA_3R_DESC_BASE<"hadd_s.d", int_mips_hadd_s_d,
                                       NoItinerary, MSA128D, MSA128W>;

class HADD_U_H_DESC : MSA_3R_DESC_BASE<"hadd_u.h", int_mips_hadd_u_h,
                                       NoItinerary, MSA128H, MSA128B>;
class HADD_U_W_DESC : MSA_3R_DESC_BASE<"hadd_u.w", int_mips_hadd_u_w,
                                       NoItinerary, MSA128W, MSA128H>;
class HADD_U_D_DESC : MSA_3R_DESC_BASE<"hadd_u.d", int_mips_hadd_u_d,
                                       NoItinerary, MSA128D, MSA128W>;

class HSUB_S_H_DESC : MSA_3R_DESC_BASE<"hsub_s.h", int_mips_hsub_s_h,
                                       NoItinerary, MSA128H, MSA128B>;
class HSUB_S_W_DESC : MSA_3R_DESC_BASE<"hsub_s.w", int_mips_hsub_s_w,
                                       NoItinerary, MSA128W, MSA128H>;
class HSUB_S_D_DESC : MSA_3R_DESC_BASE<"hsub_s.d", int_mips_hsub_s_d,
                                       NoItinerary, MSA128D, MSA128W>;

class HSUB_U_H_DESC : MSA_3R_DESC_BASE<"hsub_u.h", int_mips_hsub_u_h,
                                       NoItinerary, MSA128H, MSA128B>;
class HSUB_U_W_DESC : MSA_3R_DESC_BASE<"hsub_u.w", int_mips_hsub_u_w,
                                       NoItinerary, MSA128W, MSA128H>;
class HSUB_U_D_DESC : MSA_3R_DESC_BASE<"hsub_u.d", int_mips_hsub_u_d,
                                       NoItinerary, MSA128D, MSA128W>;

class ILVEV_B_DESC : MSA_3R_DESC_BASE<"ilvev.b", int_mips_ilvev_b, NoItinerary,
                                      MSA128B, MSA128B>;
class ILVEV_H_DESC : MSA_3R_DESC_BASE<"ilvev.h", int_mips_ilvev_h, NoItinerary,
                                      MSA128H, MSA128H>;
class ILVEV_W_DESC : MSA_3R_DESC_BASE<"ilvev.w", int_mips_ilvev_w, NoItinerary,
                                      MSA128W, MSA128W>;
class ILVEV_D_DESC : MSA_3R_DESC_BASE<"ilvev.d", int_mips_ilvev_d, NoItinerary,
                                      MSA128D, MSA128D>;

class ILVL_B_DESC : MSA_3R_DESC_BASE<"ilvl.b", int_mips_ilvl_b, NoItinerary,
                                     MSA128B, MSA128B>;
class ILVL_H_DESC : MSA_3R_DESC_BASE<"ilvl.h", int_mips_ilvl_h, NoItinerary,
                                     MSA128H, MSA128H>;
class ILVL_W_DESC : MSA_3R_DESC_BASE<"ilvl.w", int_mips_ilvl_w, NoItinerary,
                                     MSA128W, MSA128W>;
class ILVL_D_DESC : MSA_3R_DESC_BASE<"ilvl.d", int_mips_ilvl_d, NoItinerary,
                                     MSA128D, MSA128D>;

class ILVOD_B_DESC : MSA_3R_DESC_BASE<"ilvod.b", int_mips_ilvod_b, NoItinerary,
                                      MSA128B, MSA128B>;
class ILVOD_H_DESC : MSA_3R_DESC_BASE<"ilvod.h", int_mips_ilvod_h, NoItinerary,
                                      MSA128H, MSA128H>;
class ILVOD_W_DESC : MSA_3R_DESC_BASE<"ilvod.w", int_mips_ilvod_w, NoItinerary,
                                      MSA128W, MSA128W>;
class ILVOD_D_DESC : MSA_3R_DESC_BASE<"ilvod.d", int_mips_ilvod_d, NoItinerary,
                                      MSA128D, MSA128D>;

class ILVR_B_DESC : MSA_3R_DESC_BASE<"ilvr.b", int_mips_ilvr_b, NoItinerary,
                                     MSA128B, MSA128B>;
class ILVR_H_DESC : MSA_3R_DESC_BASE<"ilvr.h", int_mips_ilvr_h, NoItinerary,
                                     MSA128H, MSA128H>;
class ILVR_W_DESC : MSA_3R_DESC_BASE<"ilvr.w", int_mips_ilvr_w, NoItinerary,
                                     MSA128W, MSA128W>;
class ILVR_D_DESC : MSA_3R_DESC_BASE<"ilvr.d", int_mips_ilvr_d, NoItinerary,
                                     MSA128D, MSA128D>;

class INSERT_B_DESC : MSA_INSERT_DESC_BASE<"insert.b", int_mips_insert_b,
                                           NoItinerary, MSA128B, GPR32>;
class INSERT_H_DESC : MSA_INSERT_DESC_BASE<"insert.h", int_mips_insert_h,
                                           NoItinerary, MSA128H, GPR32>;
class INSERT_W_DESC : MSA_INSERT_DESC_BASE<"insert.w", int_mips_insert_w,
                                           NoItinerary, MSA128W, GPR32>;

class INSVE_B_DESC : MSA_INSVE_DESC_BASE<"insve.b", int_mips_insve_b,
                                         NoItinerary, MSA128B, MSA128B>;
class INSVE_H_DESC : MSA_INSVE_DESC_BASE<"insve.h", int_mips_insve_h,
                                         NoItinerary, MSA128H, MSA128H>;
class INSVE_W_DESC : MSA_INSVE_DESC_BASE<"insve.w", int_mips_insve_w,
                                         NoItinerary, MSA128W, MSA128W>;
class INSVE_D_DESC : MSA_INSVE_DESC_BASE<"insve.d", int_mips_insve_d,
                                         NoItinerary, MSA128D, MSA128D>;

class LD_DESC_BASE<string instr_asm, SDPatternOperator OpNode,
                   ValueType TyNode, InstrItinClass itin, RegisterClass RCWD,
                   Operand MemOpnd = mem, ComplexPattern Addr = addrRegImm> {
  dag OutOperandList = (outs RCWD:$wd);
  dag InOperandList = (ins MemOpnd:$addr);
  string AsmString = !strconcat(instr_asm, "\t$wd, $addr");
  list<dag> Pattern = [(set RCWD:$wd, (TyNode (OpNode Addr:$addr)))];
  InstrItinClass Itinerary = itin;
}

class LD_B_DESC : LD_DESC_BASE<"ld.b", load, v16i8, NoItinerary, MSA128B>;
class LD_H_DESC : LD_DESC_BASE<"ld.h", load, v8i16, NoItinerary, MSA128H>;
class LD_W_DESC : LD_DESC_BASE<"ld.w", load, v4i32, NoItinerary, MSA128W>;
class LD_D_DESC : LD_DESC_BASE<"ld.d", load, v2i64, NoItinerary, MSA128D>;

class LDI_B_DESC : MSA_I10_DESC_BASE<"ldi.b", int_mips_ldi_b,
                                     NoItinerary, MSA128B>;
class LDI_H_DESC : MSA_I10_DESC_BASE<"ldi.h", int_mips_ldi_h,
                                     NoItinerary, MSA128H>;
class LDI_W_DESC : MSA_I10_DESC_BASE<"ldi.w", int_mips_ldi_w,
                                     NoItinerary, MSA128W>;
class LDI_D_DESC : MSA_I10_DESC_BASE<"ldi.d", int_mips_ldi_d,
                                     NoItinerary, MSA128D>;

class LDX_DESC_BASE<string instr_asm, SDPatternOperator OpNode,
                    ValueType TyNode, InstrItinClass itin, RegisterClass RCWD,
                    Operand MemOpnd = mem, ComplexPattern Addr = addrRegReg> {
  dag OutOperandList = (outs RCWD:$wd);
  dag InOperandList = (ins MemOpnd:$addr);
  string AsmString = !strconcat(instr_asm, "\t$wd, $addr");
  list<dag> Pattern = [(set RCWD:$wd, (TyNode (OpNode Addr:$addr)))];
  InstrItinClass Itinerary = itin;
}

class LDX_B_DESC : LDX_DESC_BASE<"ldx.b", load, v16i8, NoItinerary, MSA128B>;
class LDX_H_DESC : LDX_DESC_BASE<"ldx.h", load, v8i16, NoItinerary, MSA128H>;
class LDX_W_DESC : LDX_DESC_BASE<"ldx.w", load, v4i32, NoItinerary, MSA128W>;
class LDX_D_DESC : LDX_DESC_BASE<"ldx.d", load, v2i64, NoItinerary, MSA128D>;

class MADD_Q_H_DESC : MSA_3RF_4RF_DESC_BASE<"madd_q.h", int_mips_madd_q_h,
                                            NoItinerary, MSA128H, MSA128H>;
class MADD_Q_W_DESC : MSA_3RF_4RF_DESC_BASE<"madd_q.w", int_mips_madd_q_w,
                                            NoItinerary, MSA128W, MSA128W>;

class MADDR_Q_H_DESC : MSA_3RF_4RF_DESC_BASE<"maddr_q.h", int_mips_maddr_q_h,
                                             NoItinerary, MSA128H, MSA128H>;
class MADDR_Q_W_DESC : MSA_3RF_4RF_DESC_BASE<"maddr_q.w", int_mips_maddr_q_w,
                                             NoItinerary, MSA128W, MSA128W>;

class MADDV_B_DESC : MSA_3R_4R_DESC_BASE<"maddv.b", int_mips_maddv_b,
                                         NoItinerary, MSA128B, MSA128B>;
class MADDV_H_DESC : MSA_3R_4R_DESC_BASE<"maddv.h", int_mips_maddv_h,
                                         NoItinerary, MSA128H, MSA128H>;
class MADDV_W_DESC : MSA_3R_4R_DESC_BASE<"maddv.w", int_mips_maddv_w,
                                         NoItinerary, MSA128W, MSA128W>;
class MADDV_D_DESC : MSA_3R_4R_DESC_BASE<"maddv.d", int_mips_maddv_d,
                                         NoItinerary, MSA128D, MSA128D>;

class MAX_A_B_DESC : MSA_3R_DESC_BASE<"max_a.b", int_mips_max_a_b, NoItinerary,
                                      MSA128B, MSA128B>;
class MAX_A_H_DESC : MSA_3R_DESC_BASE<"max_a.h", int_mips_max_a_h, NoItinerary,
                                      MSA128H, MSA128H>;
class MAX_A_W_DESC : MSA_3R_DESC_BASE<"max_a.w", int_mips_max_a_w, NoItinerary,
                                      MSA128W, MSA128W>;
class MAX_A_D_DESC : MSA_3R_DESC_BASE<"max_a.d", int_mips_max_a_d, NoItinerary,
                                      MSA128D, MSA128D>;

class MAX_S_B_DESC : MSA_3R_DESC_BASE<"max_s.b", int_mips_max_s_b, NoItinerary,
                                      MSA128B, MSA128B>;
class MAX_S_H_DESC : MSA_3R_DESC_BASE<"max_s.h", int_mips_max_s_h, NoItinerary,
                                      MSA128H, MSA128H>;
class MAX_S_W_DESC : MSA_3R_DESC_BASE<"max_s.w", int_mips_max_s_w, NoItinerary,
                                      MSA128W, MSA128W>;
class MAX_S_D_DESC : MSA_3R_DESC_BASE<"max_s.d", int_mips_max_s_d, NoItinerary,
                                      MSA128D, MSA128D>;

class MAX_U_B_DESC : MSA_3R_DESC_BASE<"max_u.b", int_mips_max_u_b, NoItinerary,
                                      MSA128B, MSA128B>;
class MAX_U_H_DESC : MSA_3R_DESC_BASE<"max_u.h", int_mips_max_u_h, NoItinerary,
                                      MSA128H, MSA128H>;
class MAX_U_W_DESC : MSA_3R_DESC_BASE<"max_u.w", int_mips_max_u_w, NoItinerary,
                                      MSA128W, MSA128W>;
class MAX_U_D_DESC : MSA_3R_DESC_BASE<"max_u.d", int_mips_max_u_d, NoItinerary,
                                      MSA128D, MSA128D>;

class MAXI_S_B_DESC : MSA_I5_DESC_BASE<"maxi_s.b", int_mips_maxi_s_b,
                                       NoItinerary, MSA128B, MSA128B>;
class MAXI_S_H_DESC : MSA_I5_DESC_BASE<"maxi_s.h", int_mips_maxi_s_h,
                                       NoItinerary, MSA128H, MSA128H>;
class MAXI_S_W_DESC : MSA_I5_DESC_BASE<"maxi_s.w", int_mips_maxi_s_w,
                                       NoItinerary, MSA128W, MSA128W>;
class MAXI_S_D_DESC : MSA_I5_DESC_BASE<"maxi_s.d", int_mips_maxi_s_d,
                                       NoItinerary, MSA128D, MSA128D>;

class MAXI_U_B_DESC : MSA_I5_DESC_BASE<"maxi_u.b", int_mips_maxi_u_b,
                                       NoItinerary, MSA128B, MSA128B>;
class MAXI_U_H_DESC : MSA_I5_DESC_BASE<"maxi_u.h", int_mips_maxi_u_h,
                                       NoItinerary, MSA128H, MSA128H>;
class MAXI_U_W_DESC : MSA_I5_DESC_BASE<"maxi_u.w", int_mips_maxi_u_w,
                                       NoItinerary, MSA128W, MSA128W>;
class MAXI_U_D_DESC : MSA_I5_DESC_BASE<"maxi_u.d", int_mips_maxi_u_d,
                                       NoItinerary, MSA128D, MSA128D>;

class MIN_A_B_DESC : MSA_3R_DESC_BASE<"min_a.b", int_mips_min_a_b, NoItinerary,
                                      MSA128B, MSA128B>;
class MIN_A_H_DESC : MSA_3R_DESC_BASE<"min_a.h", int_mips_min_a_h, NoItinerary,
                                      MSA128H, MSA128H>;
class MIN_A_W_DESC : MSA_3R_DESC_BASE<"min_a.w", int_mips_min_a_w, NoItinerary,
                                      MSA128W, MSA128W>;
class MIN_A_D_DESC : MSA_3R_DESC_BASE<"min_a.d", int_mips_min_a_d, NoItinerary,
                                      MSA128D, MSA128D>;

class MIN_S_B_DESC : MSA_3R_DESC_BASE<"min_s.b", int_mips_min_s_b, NoItinerary,
                                      MSA128B, MSA128B>;
class MIN_S_H_DESC : MSA_3R_DESC_BASE<"min_s.h", int_mips_min_s_h, NoItinerary,
                                      MSA128H, MSA128H>;
class MIN_S_W_DESC : MSA_3R_DESC_BASE<"min_s.w", int_mips_min_s_w, NoItinerary,
                                      MSA128W, MSA128W>;
class MIN_S_D_DESC : MSA_3R_DESC_BASE<"min_s.d", int_mips_min_s_d, NoItinerary,
                                      MSA128D, MSA128D>;

class MIN_U_B_DESC : MSA_3R_DESC_BASE<"min_u.b", int_mips_min_u_b, NoItinerary,
                                      MSA128B, MSA128B>;
class MIN_U_H_DESC : MSA_3R_DESC_BASE<"min_u.h", int_mips_min_u_h, NoItinerary,
                                      MSA128H, MSA128H>;
class MIN_U_W_DESC : MSA_3R_DESC_BASE<"min_u.w", int_mips_min_u_w, NoItinerary,
                                      MSA128W, MSA128W>;
class MIN_U_D_DESC : MSA_3R_DESC_BASE<"min_u.d", int_mips_min_u_d, NoItinerary,
                                      MSA128D, MSA128D>;

class MINI_S_B_DESC : MSA_I5_DESC_BASE<"mini_s.b", int_mips_mini_s_b,
                                       NoItinerary, MSA128B, MSA128B>;
class MINI_S_H_DESC : MSA_I5_DESC_BASE<"mini_s.h", int_mips_mini_s_h,
                                       NoItinerary, MSA128H, MSA128H>;
class MINI_S_W_DESC : MSA_I5_DESC_BASE<"mini_s.w", int_mips_mini_s_w,
                                       NoItinerary, MSA128W, MSA128W>;
class MINI_S_D_DESC : MSA_I5_DESC_BASE<"mini_s.d", int_mips_mini_s_d,
                                       NoItinerary, MSA128D, MSA128D>;

class MINI_U_B_DESC : MSA_I5_DESC_BASE<"mini_u.b", int_mips_mini_u_b,
                                       NoItinerary, MSA128B, MSA128B>;
class MINI_U_H_DESC : MSA_I5_DESC_BASE<"mini_u.h", int_mips_mini_u_h,
                                       NoItinerary, MSA128H, MSA128H>;
class MINI_U_W_DESC : MSA_I5_DESC_BASE<"mini_u.w", int_mips_mini_u_w,
                                       NoItinerary, MSA128W, MSA128W>;
class MINI_U_D_DESC : MSA_I5_DESC_BASE<"mini_u.d", int_mips_mini_u_d,
                                       NoItinerary, MSA128D, MSA128D>;

class MOD_S_B_DESC : MSA_3R_DESC_BASE<"mod_s.b", int_mips_mod_s_b, NoItinerary,
                                      MSA128B, MSA128B>;
class MOD_S_H_DESC : MSA_3R_DESC_BASE<"mod_s.h", int_mips_mod_s_h, NoItinerary,
                                      MSA128H, MSA128H>;
class MOD_S_W_DESC : MSA_3R_DESC_BASE<"mod_s.w", int_mips_mod_s_w, NoItinerary,
                                      MSA128W, MSA128W>;
class MOD_S_D_DESC : MSA_3R_DESC_BASE<"mod_s.d", int_mips_mod_s_d, NoItinerary,
                                      MSA128D, MSA128D>;

class MOD_U_B_DESC : MSA_3R_DESC_BASE<"mod_u.b", int_mips_mod_u_b, NoItinerary,
                                      MSA128B, MSA128B>;
class MOD_U_H_DESC : MSA_3R_DESC_BASE<"mod_u.h", int_mips_mod_u_h, NoItinerary,
                                      MSA128H, MSA128H>;
class MOD_U_W_DESC : MSA_3R_DESC_BASE<"mod_u.w", int_mips_mod_u_w, NoItinerary,
                                      MSA128W, MSA128W>;
class MOD_U_D_DESC : MSA_3R_DESC_BASE<"mod_u.d", int_mips_mod_u_d, NoItinerary,
                                      MSA128D, MSA128D>;

class MOVE_V_DESC {
  dag OutOperandList = (outs MSA128B:$wd);
  dag InOperandList = (ins MSA128B:$ws);
  string AsmString = "move.v\t$wd, $ws";
  list<dag> Pattern = [];
  InstrItinClass Itinerary = NoItinerary;
}

class MSUB_Q_H_DESC : MSA_3RF_4RF_DESC_BASE<"msub_q.h", int_mips_msub_q_h,
                                            NoItinerary, MSA128H, MSA128H>;
class MSUB_Q_W_DESC : MSA_3RF_4RF_DESC_BASE<"msub_q.w", int_mips_msub_q_w,
                                            NoItinerary, MSA128W, MSA128W>;

class MSUBR_Q_H_DESC : MSA_3RF_4RF_DESC_BASE<"msubr_q.h", int_mips_msubr_q_h,
                                             NoItinerary, MSA128H, MSA128H>;
class MSUBR_Q_W_DESC : MSA_3RF_4RF_DESC_BASE<"msubr_q.w", int_mips_msubr_q_w,
                                             NoItinerary, MSA128W, MSA128W>;

class MSUBV_B_DESC : MSA_3R_4R_DESC_BASE<"msubv.b", int_mips_msubv_b,
                                         NoItinerary, MSA128B, MSA128B>;
class MSUBV_H_DESC : MSA_3R_4R_DESC_BASE<"msubv.h", int_mips_msubv_h,
                                         NoItinerary, MSA128H, MSA128H>;
class MSUBV_W_DESC : MSA_3R_4R_DESC_BASE<"msubv.w", int_mips_msubv_w,
                                         NoItinerary, MSA128W, MSA128W>;
class MSUBV_D_DESC : MSA_3R_4R_DESC_BASE<"msubv.d", int_mips_msubv_d,
                                         NoItinerary, MSA128D, MSA128D>;

class MUL_Q_H_DESC : MSA_3RF_DESC_BASE<"mul_q.h", int_mips_mul_q_h,
                                       NoItinerary, MSA128H, MSA128H>;
class MUL_Q_W_DESC : MSA_3RF_DESC_BASE<"mul_q.w", int_mips_mul_q_w,
                                       NoItinerary, MSA128W, MSA128W>;

class MULR_Q_H_DESC : MSA_3RF_DESC_BASE<"mulr_q.h", int_mips_mulr_q_h,
                                        NoItinerary, MSA128H, MSA128H>;
class MULR_Q_W_DESC : MSA_3RF_DESC_BASE<"mulr_q.w", int_mips_mulr_q_w,
                                        NoItinerary, MSA128W, MSA128W>;

class MULV_B_DESC : MSA_3R_DESC_BASE<"mulv.b", int_mips_mulv_b,
                                     NoItinerary, MSA128B, MSA128B>;
class MULV_H_DESC : MSA_3R_DESC_BASE<"mulv.h", int_mips_mulv_h,
                                     NoItinerary, MSA128H, MSA128H>;
class MULV_W_DESC : MSA_3R_DESC_BASE<"mulv.w", int_mips_mulv_w,
                                     NoItinerary, MSA128W, MSA128W>;
class MULV_D_DESC : MSA_3R_DESC_BASE<"mulv.d", int_mips_mulv_d,
                                     NoItinerary, MSA128D, MSA128D>;

class NLOC_B_DESC : MSA_2R_DESC_BASE<"nloc.b", int_mips_nloc_b,
                                     NoItinerary, MSA128B, MSA128B>;
class NLOC_H_DESC : MSA_2R_DESC_BASE<"nloc.h", int_mips_nloc_h,
                                     NoItinerary, MSA128H, MSA128H>;
class NLOC_W_DESC : MSA_2R_DESC_BASE<"nloc.w", int_mips_nloc_w,
                                     NoItinerary, MSA128W, MSA128W>;
class NLOC_D_DESC : MSA_2R_DESC_BASE<"nloc.d", int_mips_nloc_d,
                                     NoItinerary, MSA128D, MSA128D>;

class NLZC_B_DESC : MSA_2R_DESC_BASE<"nlzc.b", int_mips_nlzc_b,
                                     NoItinerary, MSA128B, MSA128B>;
class NLZC_H_DESC : MSA_2R_DESC_BASE<"nlzc.h", int_mips_nlzc_h,
                                     NoItinerary, MSA128H, MSA128H>;
class NLZC_W_DESC : MSA_2R_DESC_BASE<"nlzc.w", int_mips_nlzc_w,
                                     NoItinerary, MSA128W, MSA128W>;
class NLZC_D_DESC : MSA_2R_DESC_BASE<"nlzc.d", int_mips_nlzc_d,
                                     NoItinerary, MSA128D, MSA128D>;

class NOR_V_DESC : MSA_VEC_DESC_BASE<"nor.v", int_mips_nor_v, NoItinerary,
                                     MSA128B, MSA128B>;

class NORI_B_DESC : MSA_I8_DESC_BASE<"nori.b", int_mips_nori_b, NoItinerary,
                                     MSA128B, MSA128B>;

class OR_V_DESC : MSA_VEC_DESC_BASE<"or.v", int_mips_or_v, NoItinerary,
                                    MSA128B, MSA128B>;

class ORI_B_DESC : MSA_I8_DESC_BASE<"ori.b", int_mips_ori_b, NoItinerary,
                                    MSA128B, MSA128B>;

class PCKEV_B_DESC : MSA_3R_DESC_BASE<"pckev.b", int_mips_pckev_b, NoItinerary,
                                      MSA128B, MSA128B>;
class PCKEV_H_DESC : MSA_3R_DESC_BASE<"pckev.h", int_mips_pckev_h, NoItinerary,
                                      MSA128H, MSA128H>;
class PCKEV_W_DESC : MSA_3R_DESC_BASE<"pckev.w", int_mips_pckev_w, NoItinerary,
                                      MSA128W, MSA128W>;
class PCKEV_D_DESC : MSA_3R_DESC_BASE<"pckev.d", int_mips_pckev_d, NoItinerary,
                                      MSA128D, MSA128D>;

class PCKOD_B_DESC : MSA_3R_DESC_BASE<"pckod.b", int_mips_pckod_b, NoItinerary,
                                      MSA128B, MSA128B>;
class PCKOD_H_DESC : MSA_3R_DESC_BASE<"pckod.h", int_mips_pckod_h, NoItinerary,
                                      MSA128H, MSA128H>;
class PCKOD_W_DESC : MSA_3R_DESC_BASE<"pckod.w", int_mips_pckod_w, NoItinerary,
                                      MSA128W, MSA128W>;
class PCKOD_D_DESC : MSA_3R_DESC_BASE<"pckod.d", int_mips_pckod_d, NoItinerary,
                                      MSA128D, MSA128D>;

class PCNT_B_DESC : MSA_2R_DESC_BASE<"pcnt.b", int_mips_pcnt_b,
                                     NoItinerary, MSA128B, MSA128B>;
class PCNT_H_DESC : MSA_2R_DESC_BASE<"pcnt.h", int_mips_pcnt_h,
                                     NoItinerary, MSA128H, MSA128H>;
class PCNT_W_DESC : MSA_2R_DESC_BASE<"pcnt.w", int_mips_pcnt_w,
                                     NoItinerary, MSA128W, MSA128W>;
class PCNT_D_DESC : MSA_2R_DESC_BASE<"pcnt.d", int_mips_pcnt_d,
                                     NoItinerary, MSA128D, MSA128D>;

class SAT_S_B_DESC : MSA_BIT_B_DESC_BASE<"sat_s.b", int_mips_sat_s_b,
                                         NoItinerary, MSA128B, MSA128B>;
class SAT_S_H_DESC : MSA_BIT_H_DESC_BASE<"sat_s.h", int_mips_sat_s_h,
                                         NoItinerary, MSA128H, MSA128H>;
class SAT_S_W_DESC : MSA_BIT_W_DESC_BASE<"sat_s.w", int_mips_sat_s_w,
                                         NoItinerary, MSA128W, MSA128W>;
class SAT_S_D_DESC : MSA_BIT_D_DESC_BASE<"sat_s.d", int_mips_sat_s_d,
                                         NoItinerary, MSA128D, MSA128D>;

class SAT_U_B_DESC : MSA_BIT_B_DESC_BASE<"sat_u.b", int_mips_sat_u_b,
                                         NoItinerary, MSA128B, MSA128B>;
class SAT_U_H_DESC : MSA_BIT_H_DESC_BASE<"sat_u.h", int_mips_sat_u_h,
                                         NoItinerary, MSA128H, MSA128H>;
class SAT_U_W_DESC : MSA_BIT_W_DESC_BASE<"sat_u.w", int_mips_sat_u_w,
                                         NoItinerary, MSA128W, MSA128W>;
class SAT_U_D_DESC : MSA_BIT_D_DESC_BASE<"sat_u.d", int_mips_sat_u_d,
                                         NoItinerary, MSA128D, MSA128D>;

class SHF_B_DESC : MSA_I8_DESC_BASE<"shf.b", int_mips_shf_b, NoItinerary,
                                    MSA128B, MSA128B>;
class SHF_H_DESC : MSA_I8_DESC_BASE<"shf.h", int_mips_shf_h, NoItinerary,
                                    MSA128H, MSA128H>;
class SHF_W_DESC : MSA_I8_DESC_BASE<"shf.w", int_mips_shf_w, NoItinerary,
                                    MSA128W, MSA128W>;

class SLD_B_DESC : MSA_3R_DESC_BASE<"sld.b", int_mips_sld_b, NoItinerary,
                                    MSA128B, MSA128B>;
class SLD_H_DESC : MSA_3R_DESC_BASE<"sld.h", int_mips_sld_h, NoItinerary,
                                    MSA128H, MSA128H>;
class SLD_W_DESC : MSA_3R_DESC_BASE<"sld.w", int_mips_sld_w, NoItinerary,
                                    MSA128W, MSA128W>;
class SLD_D_DESC : MSA_3R_DESC_BASE<"sld.d", int_mips_sld_d, NoItinerary,
                                    MSA128D, MSA128D>;

class SLDI_B_DESC : MSA_BIT_B_DESC_BASE<"sldi.b", int_mips_sldi_b,
                                        NoItinerary, MSA128B, MSA128B>;
class SLDI_H_DESC : MSA_BIT_H_DESC_BASE<"sldi.h", int_mips_sldi_h,
                                        NoItinerary, MSA128H, MSA128H>;
class SLDI_W_DESC : MSA_BIT_W_DESC_BASE<"sldi.w", int_mips_sldi_w,
                                        NoItinerary, MSA128W, MSA128W>;
class SLDI_D_DESC : MSA_BIT_D_DESC_BASE<"sldi.d", int_mips_sldi_d,
                                        NoItinerary, MSA128D, MSA128D>;

class SLL_B_DESC : MSA_3R_DESC_BASE<"sll.b", int_mips_sll_b, NoItinerary,
                                    MSA128B, MSA128B>;
class SLL_H_DESC : MSA_3R_DESC_BASE<"sll.h", int_mips_sll_h, NoItinerary,
                                    MSA128H, MSA128H>;
class SLL_W_DESC : MSA_3R_DESC_BASE<"sll.w", int_mips_sll_w, NoItinerary,
                                    MSA128W, MSA128W>;
class SLL_D_DESC : MSA_3R_DESC_BASE<"sll.d", int_mips_sll_d, NoItinerary,
                                    MSA128D, MSA128D>;

class SLLI_B_DESC : MSA_BIT_B_DESC_BASE<"slli.b", int_mips_slli_b,
                                        NoItinerary, MSA128B, MSA128B>;
class SLLI_H_DESC : MSA_BIT_H_DESC_BASE<"slli.h", int_mips_slli_h,
                                        NoItinerary, MSA128H, MSA128H>;
class SLLI_W_DESC : MSA_BIT_W_DESC_BASE<"slli.w", int_mips_slli_w,
                                        NoItinerary, MSA128W, MSA128W>;
class SLLI_D_DESC : MSA_BIT_D_DESC_BASE<"slli.d", int_mips_slli_d,
                                        NoItinerary, MSA128D, MSA128D>;

class SPLAT_B_DESC : MSA_3R_DESC_BASE<"splat.b", int_mips_splat_b, NoItinerary,
                                      MSA128B, MSA128B, GPR32>;
class SPLAT_H_DESC : MSA_3R_DESC_BASE<"splat.h", int_mips_splat_h, NoItinerary,
                                      MSA128H, MSA128H, GPR32>;
class SPLAT_W_DESC : MSA_3R_DESC_BASE<"splat.w", int_mips_splat_w, NoItinerary,
                                      MSA128W, MSA128W, GPR32>;
class SPLAT_D_DESC : MSA_3R_DESC_BASE<"splat.d", int_mips_splat_d, NoItinerary,
                                      MSA128D, MSA128D, GPR32>;

class SPLATI_B_DESC : MSA_BIT_B_DESC_BASE<"splati.b", int_mips_splati_b,
                                          NoItinerary, MSA128B, MSA128B>;
class SPLATI_H_DESC : MSA_BIT_H_DESC_BASE<"splati.h", int_mips_splati_h,
                                          NoItinerary, MSA128H, MSA128H>;
class SPLATI_W_DESC : MSA_BIT_W_DESC_BASE<"splati.w", int_mips_splati_w,
                                          NoItinerary, MSA128W, MSA128W>;
class SPLATI_D_DESC : MSA_BIT_D_DESC_BASE<"splati.d", int_mips_splati_d,
                                          NoItinerary, MSA128D, MSA128D>;

class SRA_B_DESC : MSA_3R_DESC_BASE<"sra.b", int_mips_sra_b, NoItinerary,
                                    MSA128B, MSA128B>;
class SRA_H_DESC : MSA_3R_DESC_BASE<"sra.h", int_mips_sra_h, NoItinerary,
                                    MSA128H, MSA128H>;
class SRA_W_DESC : MSA_3R_DESC_BASE<"sra.w", int_mips_sra_w, NoItinerary,
                                    MSA128W, MSA128W>;
class SRA_D_DESC : MSA_3R_DESC_BASE<"sra.d", int_mips_sra_d, NoItinerary,
                                    MSA128D, MSA128D>;

class SRAI_B_DESC : MSA_BIT_B_DESC_BASE<"srai.b", int_mips_srai_b,
                                        NoItinerary, MSA128B, MSA128B>;
class SRAI_H_DESC : MSA_BIT_H_DESC_BASE<"srai.h", int_mips_srai_h,
                                        NoItinerary, MSA128H, MSA128H>;
class SRAI_W_DESC : MSA_BIT_W_DESC_BASE<"srai.w", int_mips_srai_w,
                                        NoItinerary, MSA128W, MSA128W>;
class SRAI_D_DESC : MSA_BIT_D_DESC_BASE<"srai.d", int_mips_srai_d,
                                        NoItinerary, MSA128D, MSA128D>;

class SRAR_B_DESC : MSA_3R_DESC_BASE<"srar.b", int_mips_srar_b, NoItinerary,
                                     MSA128B, MSA128B>;
class SRAR_H_DESC : MSA_3R_DESC_BASE<"srar.h", int_mips_srar_h, NoItinerary,
                                     MSA128H, MSA128H>;
class SRAR_W_DESC : MSA_3R_DESC_BASE<"srar.w", int_mips_srar_w, NoItinerary,
                                     MSA128W, MSA128W>;
class SRAR_D_DESC : MSA_3R_DESC_BASE<"srar.d", int_mips_srar_d, NoItinerary,
                                     MSA128D, MSA128D>;

class SRARI_B_DESC : MSA_BIT_B_DESC_BASE<"srari.b", int_mips_srari_b,
                                         NoItinerary, MSA128B, MSA128B>;
class SRARI_H_DESC : MSA_BIT_H_DESC_BASE<"srari.h", int_mips_srari_h,
                                         NoItinerary, MSA128H, MSA128H>;
class SRARI_W_DESC : MSA_BIT_W_DESC_BASE<"srari.w", int_mips_srari_w,
                                         NoItinerary, MSA128W, MSA128W>;
class SRARI_D_DESC : MSA_BIT_D_DESC_BASE<"srari.d", int_mips_srari_d,
                                         NoItinerary, MSA128D, MSA128D>;

class SRL_B_DESC : MSA_3R_DESC_BASE<"srl.b", int_mips_srl_b, NoItinerary,
                                    MSA128B, MSA128B>;
class SRL_H_DESC : MSA_3R_DESC_BASE<"srl.h", int_mips_srl_h, NoItinerary,
                                    MSA128H, MSA128H>;
class SRL_W_DESC : MSA_3R_DESC_BASE<"srl.w", int_mips_srl_w, NoItinerary,
                                    MSA128W, MSA128W>;
class SRL_D_DESC : MSA_3R_DESC_BASE<"srl.d", int_mips_srl_d, NoItinerary,
                                    MSA128D, MSA128D>;

class SRLI_B_DESC : MSA_BIT_B_DESC_BASE<"srli.b", int_mips_srli_b,
                                        NoItinerary, MSA128B, MSA128B>;
class SRLI_H_DESC : MSA_BIT_H_DESC_BASE<"srli.h", int_mips_srli_h,
                                        NoItinerary, MSA128H, MSA128H>;
class SRLI_W_DESC : MSA_BIT_W_DESC_BASE<"srli.w", int_mips_srli_w,
                                        NoItinerary, MSA128W, MSA128W>;
class SRLI_D_DESC : MSA_BIT_D_DESC_BASE<"srli.d", int_mips_srli_d,
                                        NoItinerary, MSA128D, MSA128D>;

class SRLR_B_DESC : MSA_3R_DESC_BASE<"srlr.b", int_mips_srlr_b, NoItinerary,
                                     MSA128B, MSA128B>;
class SRLR_H_DESC : MSA_3R_DESC_BASE<"srlr.h", int_mips_srlr_h, NoItinerary,
                                     MSA128H, MSA128H>;
class SRLR_W_DESC : MSA_3R_DESC_BASE<"srlr.w", int_mips_srlr_w, NoItinerary,
                                     MSA128W, MSA128W>;
class SRLR_D_DESC : MSA_3R_DESC_BASE<"srlr.d", int_mips_srlr_d, NoItinerary,
                                     MSA128D, MSA128D>;

class SRLRI_B_DESC : MSA_BIT_B_DESC_BASE<"srlri.b", int_mips_srlri_b,
                                         NoItinerary, MSA128B, MSA128B>;
class SRLRI_H_DESC : MSA_BIT_H_DESC_BASE<"srlri.h", int_mips_srlri_h,
                                         NoItinerary, MSA128H, MSA128H>;
class SRLRI_W_DESC : MSA_BIT_W_DESC_BASE<"srlri.w", int_mips_srlri_w,
                                         NoItinerary, MSA128W, MSA128W>;
class SRLRI_D_DESC : MSA_BIT_D_DESC_BASE<"srlri.d", int_mips_srlri_d,
                                         NoItinerary, MSA128D, MSA128D>;

class ST_DESC_BASE<string instr_asm, SDPatternOperator OpNode,
                   ValueType TyNode, InstrItinClass itin, RegisterClass RCWD,
                   Operand MemOpnd = mem, ComplexPattern Addr = addrRegImm> {
  dag OutOperandList = (outs);
  dag InOperandList = (ins RCWD:$wd, MemOpnd:$addr);
  string AsmString = !strconcat(instr_asm, "\t$wd, $addr");
  list<dag> Pattern = [(OpNode (TyNode RCWD:$wd), Addr:$addr)];
  InstrItinClass Itinerary = itin;
}

class ST_B_DESC : ST_DESC_BASE<"st.b", store, v16i8, NoItinerary, MSA128B>;
class ST_H_DESC : ST_DESC_BASE<"st.h", store, v8i16, NoItinerary, MSA128H>;
class ST_W_DESC : ST_DESC_BASE<"st.w", store, v4i32, NoItinerary, MSA128W>;
class ST_D_DESC : ST_DESC_BASE<"st.d", store, v2i64, NoItinerary, MSA128D>;

class STX_DESC_BASE<string instr_asm, SDPatternOperator OpNode,
                    ValueType TyNode, InstrItinClass itin, RegisterClass RCWD,
                    Operand MemOpnd = mem, ComplexPattern Addr = addrRegReg> {
  dag OutOperandList = (outs);
  dag InOperandList = (ins RCWD:$wd, MemOpnd:$addr);
  string AsmString = !strconcat(instr_asm, "\t$wd, $addr");
  list<dag> Pattern = [(OpNode (TyNode RCWD:$wd), Addr:$addr)];
  InstrItinClass Itinerary = itin;
}

class STX_B_DESC : STX_DESC_BASE<"stx.b", store, v16i8, NoItinerary, MSA128B>;
class STX_H_DESC : STX_DESC_BASE<"stx.h", store, v8i16, NoItinerary, MSA128H>;
class STX_W_DESC : STX_DESC_BASE<"stx.w", store, v4i32, NoItinerary, MSA128W>;
class STX_D_DESC : STX_DESC_BASE<"stx.d", store, v2i64, NoItinerary, MSA128D>;

class SUBS_S_B_DESC : MSA_3R_DESC_BASE<"subs_s.b", int_mips_subs_s_b,
                                       NoItinerary, MSA128B, MSA128B>;
class SUBS_S_H_DESC : MSA_3R_DESC_BASE<"subs_s.h", int_mips_subs_s_h,
                                       NoItinerary, MSA128H, MSA128H>;
class SUBS_S_W_DESC : MSA_3R_DESC_BASE<"subs_s.w", int_mips_subs_s_w,
                                       NoItinerary, MSA128W, MSA128W>;
class SUBS_S_D_DESC : MSA_3R_DESC_BASE<"subs_s.d", int_mips_subs_s_d,
                                       NoItinerary, MSA128D, MSA128D>;

class SUBS_U_B_DESC : MSA_3R_DESC_BASE<"subs_u.b", int_mips_subs_u_b,
                                       NoItinerary, MSA128B, MSA128B>;
class SUBS_U_H_DESC : MSA_3R_DESC_BASE<"subs_u.h", int_mips_subs_u_h,
                                       NoItinerary, MSA128H, MSA128H>;
class SUBS_U_W_DESC : MSA_3R_DESC_BASE<"subs_u.w", int_mips_subs_u_w,
                                       NoItinerary, MSA128W, MSA128W>;
class SUBS_U_D_DESC : MSA_3R_DESC_BASE<"subs_u.d", int_mips_subs_u_d,
                                       NoItinerary, MSA128D, MSA128D>;

class SUBSUS_U_B_DESC : MSA_3R_DESC_BASE<"subsus_u.b", int_mips_subsus_u_b,
                                         NoItinerary, MSA128B, MSA128B>;
class SUBSUS_U_H_DESC : MSA_3R_DESC_BASE<"subsus_u.h", int_mips_subsus_u_h,
                                         NoItinerary, MSA128H, MSA128H>;
class SUBSUS_U_W_DESC : MSA_3R_DESC_BASE<"subsus_u.w", int_mips_subsus_u_w,
                                         NoItinerary, MSA128W, MSA128W>;
class SUBSUS_U_D_DESC : MSA_3R_DESC_BASE<"subsus_u.d", int_mips_subsus_u_d,
                                         NoItinerary, MSA128D, MSA128D>;

class SUBSUU_S_B_DESC : MSA_3R_DESC_BASE<"subsuu_s.b", int_mips_subsuu_s_b,
                                         NoItinerary, MSA128B, MSA128B>;
class SUBSUU_S_H_DESC : MSA_3R_DESC_BASE<"subsuu_s.h", int_mips_subsuu_s_h,
                                         NoItinerary, MSA128H, MSA128H>;
class SUBSUU_S_W_DESC : MSA_3R_DESC_BASE<"subsuu_s.w", int_mips_subsuu_s_w,
                                         NoItinerary, MSA128W, MSA128W>;
class SUBSUU_S_D_DESC : MSA_3R_DESC_BASE<"subsuu_s.d", int_mips_subsuu_s_d,
                                         NoItinerary, MSA128D, MSA128D>;

class SUBV_B_DESC : MSA_3R_DESC_BASE<"subv.b", int_mips_subv_b,
                                     NoItinerary, MSA128B, MSA128B>;
class SUBV_H_DESC : MSA_3R_DESC_BASE<"subv.h", int_mips_subv_h,
                                     NoItinerary, MSA128H, MSA128H>;
class SUBV_W_DESC : MSA_3R_DESC_BASE<"subv.w", int_mips_subv_w,
                                     NoItinerary, MSA128W, MSA128W>;
class SUBV_D_DESC : MSA_3R_DESC_BASE<"subv.d", int_mips_subv_d,
                                     NoItinerary, MSA128D, MSA128D>;

class SUBVI_B_DESC : MSA_I5_DESC_BASE<"subvi.b", int_mips_subvi_b, NoItinerary,
                                      MSA128B, MSA128B>;
class SUBVI_H_DESC : MSA_I5_DESC_BASE<"subvi.h", int_mips_subvi_h, NoItinerary,
                                      MSA128H, MSA128H>;
class SUBVI_W_DESC : MSA_I5_DESC_BASE<"subvi.w", int_mips_subvi_w, NoItinerary,
                                      MSA128W, MSA128W>;
class SUBVI_D_DESC : MSA_I5_DESC_BASE<"subvi.d", int_mips_subvi_d, NoItinerary,
                                      MSA128D, MSA128D>;

class VSHF_B_DESC : MSA_3R_DESC_BASE<"vshf.b", int_mips_vshf_b,
                                     NoItinerary, MSA128B, MSA128B>;
class VSHF_H_DESC : MSA_3R_DESC_BASE<"vshf.h", int_mips_vshf_h,
                                     NoItinerary, MSA128H, MSA128H>;
class VSHF_W_DESC : MSA_3R_DESC_BASE<"vshf.w", int_mips_vshf_w,
                                     NoItinerary, MSA128W, MSA128W>;
class VSHF_D_DESC : MSA_3R_DESC_BASE<"vshf.d", int_mips_vshf_d,
                                     NoItinerary, MSA128D, MSA128D>;

class XOR_V_DESC : MSA_VEC_DESC_BASE<"xor.v", int_mips_xor_v, NoItinerary,
                                     MSA128B, MSA128B>;

class XORI_B_DESC : MSA_I8_DESC_BASE<"xori.b", int_mips_xori_b, NoItinerary,
                                     MSA128B, MSA128B>;
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

def AND_V : AND_V_ENC, AND_V_DESC, Requires<[HasMSA]>;

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

def BMNZ_V : BMNZ_V_ENC, BMNZ_V_DESC, Requires<[HasMSA]>;

def BMNZI_B : BMNZI_B_ENC, BMNZI_B_DESC, Requires<[HasMSA]>;

def BMZ_V : BMZ_V_ENC, BMZ_V_DESC, Requires<[HasMSA]>;

def BMZI_B : BMZI_B_ENC, BMZI_B_DESC, Requires<[HasMSA]>;

def BNEG_B : BNEG_B_ENC, BNEG_B_DESC, Requires<[HasMSA]>;
def BNEG_H : BNEG_H_ENC, BNEG_H_DESC, Requires<[HasMSA]>;
def BNEG_W : BNEG_W_ENC, BNEG_W_DESC, Requires<[HasMSA]>;
def BNEG_D : BNEG_D_ENC, BNEG_D_DESC, Requires<[HasMSA]>;

def BNEGI_B : BNEGI_B_ENC, BNEGI_B_DESC, Requires<[HasMSA]>;
def BNEGI_H : BNEGI_H_ENC, BNEGI_H_DESC, Requires<[HasMSA]>;
def BNEGI_W : BNEGI_W_ENC, BNEGI_W_DESC, Requires<[HasMSA]>;
def BNEGI_D : BNEGI_D_ENC, BNEGI_D_DESC, Requires<[HasMSA]>;

def BNZ_B : BNZ_B_ENC, BNZ_B_DESC, Requires<[HasMSA]>;
def BNZ_H : BNZ_H_ENC, BNZ_H_DESC, Requires<[HasMSA]>;
def BNZ_W : BNZ_W_ENC, BNZ_W_DESC, Requires<[HasMSA]>;
def BNZ_D : BNZ_D_ENC, BNZ_D_DESC, Requires<[HasMSA]>;

def BNZ_V : BNZ_V_ENC, BNZ_V_DESC, Requires<[HasMSA]>;

def BSEL_V : BSEL_V_ENC, BSEL_V_DESC, Requires<[HasMSA]>;

def BSELI_B : BSELI_B_ENC, BSELI_B_DESC, Requires<[HasMSA]>;

def BSET_B : BSET_B_ENC, BSET_B_DESC, Requires<[HasMSA]>;
def BSET_H : BSET_H_ENC, BSET_H_DESC, Requires<[HasMSA]>;
def BSET_W : BSET_W_ENC, BSET_W_DESC, Requires<[HasMSA]>;
def BSET_D : BSET_D_ENC, BSET_D_DESC, Requires<[HasMSA]>;

def BSETI_B : BSETI_B_ENC, BSETI_B_DESC, Requires<[HasMSA]>;
def BSETI_H : BSETI_H_ENC, BSETI_H_DESC, Requires<[HasMSA]>;
def BSETI_W : BSETI_W_ENC, BSETI_W_DESC, Requires<[HasMSA]>;
def BSETI_D : BSETI_D_ENC, BSETI_D_DESC, Requires<[HasMSA]>;

def BZ_B : BZ_B_ENC, BZ_B_DESC, Requires<[HasMSA]>;
def BZ_H : BZ_H_ENC, BZ_H_DESC, Requires<[HasMSA]>;
def BZ_W : BZ_W_ENC, BZ_W_DESC, Requires<[HasMSA]>;
def BZ_D : BZ_D_ENC, BZ_D_DESC, Requires<[HasMSA]>;

def BZ_V : BZ_V_ENC, BZ_V_DESC, Requires<[HasMSA]>;

def CEQ_B : CEQ_B_ENC, CEQ_B_DESC, Requires<[HasMSA]>;
def CEQ_H : CEQ_H_ENC, CEQ_H_DESC, Requires<[HasMSA]>;
def CEQ_W : CEQ_W_ENC, CEQ_W_DESC, Requires<[HasMSA]>;
def CEQ_D : CEQ_D_ENC, CEQ_D_DESC, Requires<[HasMSA]>;

def CEQI_B : CEQI_B_ENC, CEQI_B_DESC, Requires<[HasMSA]>;
def CEQI_H : CEQI_H_ENC, CEQI_H_DESC, Requires<[HasMSA]>;
def CEQI_W : CEQI_W_ENC, CEQI_W_DESC, Requires<[HasMSA]>;
def CEQI_D : CEQI_D_ENC, CEQI_D_DESC, Requires<[HasMSA]>;

def CFCMSA : CFCMSA_ENC, CFCMSA_DESC, Requires<[HasMSA]>;

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

def CTCMSA : CTCMSA_ENC, CTCMSA_DESC, Requires<[HasMSA]>;

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

def FADD_W : FADD_W_ENC, FADD_W_DESC, Requires<[HasMSA]>;
def FADD_D : FADD_D_ENC, FADD_D_DESC, Requires<[HasMSA]>;

def FCAF_W : FCAF_W_ENC, FCAF_W_DESC, Requires<[HasMSA]>;
def FCAF_D : FCAF_D_ENC, FCAF_D_DESC, Requires<[HasMSA]>;

def FCEQ_W : FCEQ_W_ENC, FCEQ_W_DESC, Requires<[HasMSA]>;
def FCEQ_D : FCEQ_D_ENC, FCEQ_D_DESC, Requires<[HasMSA]>;

def FCLE_W : FCLE_W_ENC, FCLE_W_DESC, Requires<[HasMSA]>;
def FCLE_D : FCLE_D_ENC, FCLE_D_DESC, Requires<[HasMSA]>;

def FCLT_W : FCLT_W_ENC, FCLT_W_DESC, Requires<[HasMSA]>;
def FCLT_D : FCLT_D_ENC, FCLT_D_DESC, Requires<[HasMSA]>;

def FCLASS_W : FCLASS_W_ENC, FCLASS_W_DESC, Requires<[HasMSA]>;
def FCLASS_D : FCLASS_D_ENC, FCLASS_D_DESC, Requires<[HasMSA]>;

def FCNE_W : FCNE_W_ENC, FCNE_W_DESC, Requires<[HasMSA]>;
def FCNE_D : FCNE_D_ENC, FCNE_D_DESC, Requires<[HasMSA]>;

def FCOR_W : FCOR_W_ENC, FCOR_W_DESC, Requires<[HasMSA]>;
def FCOR_D : FCOR_D_ENC, FCOR_D_DESC, Requires<[HasMSA]>;

def FCUEQ_W : FCUEQ_W_ENC, FCUEQ_W_DESC, Requires<[HasMSA]>;
def FCUEQ_D : FCUEQ_D_ENC, FCUEQ_D_DESC, Requires<[HasMSA]>;

def FCULE_W : FCULE_W_ENC, FCULE_W_DESC, Requires<[HasMSA]>;
def FCULE_D : FCULE_D_ENC, FCULE_D_DESC, Requires<[HasMSA]>;

def FCULT_W : FCULT_W_ENC, FCULT_W_DESC, Requires<[HasMSA]>;
def FCULT_D : FCULT_D_ENC, FCULT_D_DESC, Requires<[HasMSA]>;

def FCUN_W : FCUN_W_ENC, FCUN_W_DESC, Requires<[HasMSA]>;
def FCUN_D : FCUN_D_ENC, FCUN_D_DESC, Requires<[HasMSA]>;

def FCUNE_W : FCUNE_W_ENC, FCUNE_W_DESC, Requires<[HasMSA]>;
def FCUNE_D : FCUNE_D_ENC, FCUNE_D_DESC, Requires<[HasMSA]>;

def FDIV_W : FDIV_W_ENC, FDIV_W_DESC, Requires<[HasMSA]>;
def FDIV_D : FDIV_D_ENC, FDIV_D_DESC, Requires<[HasMSA]>;

def FEXDO_H : FEXDO_H_ENC, FEXDO_H_DESC, Requires<[HasMSA]>;
def FEXDO_W : FEXDO_W_ENC, FEXDO_W_DESC, Requires<[HasMSA]>;

def FEXP2_W : FEXP2_W_ENC, FEXP2_W_DESC, Requires<[HasMSA]>;
def FEXP2_D : FEXP2_D_ENC, FEXP2_D_DESC, Requires<[HasMSA]>;

def FEXUPL_W : FEXUPL_W_ENC, FEXUPL_W_DESC, Requires<[HasMSA]>;
def FEXUPL_D : FEXUPL_D_ENC, FEXUPL_D_DESC, Requires<[HasMSA]>;

def FEXUPR_W : FEXUPR_W_ENC, FEXUPR_W_DESC, Requires<[HasMSA]>;
def FEXUPR_D : FEXUPR_D_ENC, FEXUPR_D_DESC, Requires<[HasMSA]>;

def FFINT_S_W : FFINT_S_W_ENC, FFINT_S_W_DESC, Requires<[HasMSA]>;
def FFINT_S_D : FFINT_S_D_ENC, FFINT_S_D_DESC, Requires<[HasMSA]>;

def FFINT_U_W : FFINT_U_W_ENC, FFINT_U_W_DESC, Requires<[HasMSA]>;
def FFINT_U_D : FFINT_U_D_ENC, FFINT_U_D_DESC, Requires<[HasMSA]>;

def FFQL_W : FFQL_W_ENC, FFQL_W_DESC, Requires<[HasMSA]>;
def FFQL_D : FFQL_D_ENC, FFQL_D_DESC, Requires<[HasMSA]>;

def FFQR_W : FFQR_W_ENC, FFQR_W_DESC, Requires<[HasMSA]>;
def FFQR_D : FFQR_D_ENC, FFQR_D_DESC, Requires<[HasMSA]>;

def FILL_B : FILL_B_ENC, FILL_B_DESC, Requires<[HasMSA]>;
def FILL_H : FILL_H_ENC, FILL_H_DESC, Requires<[HasMSA]>;
def FILL_W : FILL_W_ENC, FILL_W_DESC, Requires<[HasMSA]>;

def FLOG2_W : FLOG2_W_ENC, FLOG2_W_DESC, Requires<[HasMSA]>;
def FLOG2_D : FLOG2_D_ENC, FLOG2_D_DESC, Requires<[HasMSA]>;

def FMADD_W : FMADD_W_ENC, FMADD_W_DESC, Requires<[HasMSA]>;
def FMADD_D : FMADD_D_ENC, FMADD_D_DESC, Requires<[HasMSA]>;

def FMAX_W : FMAX_W_ENC, FMAX_W_DESC, Requires<[HasMSA]>;
def FMAX_D : FMAX_D_ENC, FMAX_D_DESC, Requires<[HasMSA]>;

def FMAX_A_W : FMAX_A_W_ENC, FMAX_A_W_DESC, Requires<[HasMSA]>;
def FMAX_A_D : FMAX_A_D_ENC, FMAX_A_D_DESC, Requires<[HasMSA]>;

def FMIN_W : FMIN_W_ENC, FMIN_W_DESC, Requires<[HasMSA]>;
def FMIN_D : FMIN_D_ENC, FMIN_D_DESC, Requires<[HasMSA]>;

def FMIN_A_W : FMIN_A_W_ENC, FMIN_A_W_DESC, Requires<[HasMSA]>;
def FMIN_A_D : FMIN_A_D_ENC, FMIN_A_D_DESC, Requires<[HasMSA]>;

def FMSUB_W : FMSUB_W_ENC, FMSUB_W_DESC, Requires<[HasMSA]>;
def FMSUB_D : FMSUB_D_ENC, FMSUB_D_DESC, Requires<[HasMSA]>;

def FMUL_W : FMUL_W_ENC, FMUL_W_DESC, Requires<[HasMSA]>;
def FMUL_D : FMUL_D_ENC, FMUL_D_DESC, Requires<[HasMSA]>;

def FRINT_W : FRINT_W_ENC, FRINT_W_DESC, Requires<[HasMSA]>;
def FRINT_D : FRINT_D_ENC, FRINT_D_DESC, Requires<[HasMSA]>;

def FRCP_W : FRCP_W_ENC, FRCP_W_DESC, Requires<[HasMSA]>;
def FRCP_D : FRCP_D_ENC, FRCP_D_DESC, Requires<[HasMSA]>;

def FRSQRT_W : FRSQRT_W_ENC, FRSQRT_W_DESC, Requires<[HasMSA]>;
def FRSQRT_D : FRSQRT_D_ENC, FRSQRT_D_DESC, Requires<[HasMSA]>;

def FSAF_W : FSAF_W_ENC, FSAF_W_DESC, Requires<[HasMSA]>;
def FSAF_D : FSAF_D_ENC, FSAF_D_DESC, Requires<[HasMSA]>;

def FSEQ_W : FSEQ_W_ENC, FSEQ_W_DESC, Requires<[HasMSA]>;
def FSEQ_D : FSEQ_D_ENC, FSEQ_D_DESC, Requires<[HasMSA]>;

def FSLE_W : FSLE_W_ENC, FSLE_W_DESC, Requires<[HasMSA]>;
def FSLE_D : FSLE_D_ENC, FSLE_D_DESC, Requires<[HasMSA]>;

def FSLT_W : FSLT_W_ENC, FSLT_W_DESC, Requires<[HasMSA]>;
def FSLT_D : FSLT_D_ENC, FSLT_D_DESC, Requires<[HasMSA]>;

def FSNE_W : FSNE_W_ENC, FSNE_W_DESC, Requires<[HasMSA]>;
def FSNE_D : FSNE_D_ENC, FSNE_D_DESC, Requires<[HasMSA]>;

def FSOR_W : FSOR_W_ENC, FSOR_W_DESC, Requires<[HasMSA]>;
def FSOR_D : FSOR_D_ENC, FSOR_D_DESC, Requires<[HasMSA]>;

def FSQRT_W : FSQRT_W_ENC, FSQRT_W_DESC, Requires<[HasMSA]>;
def FSQRT_D : FSQRT_D_ENC, FSQRT_D_DESC, Requires<[HasMSA]>;

def FSUB_W : FSUB_W_ENC, FSUB_W_DESC, Requires<[HasMSA]>;
def FSUB_D : FSUB_D_ENC, FSUB_D_DESC, Requires<[HasMSA]>;

def FSUEQ_W : FSUEQ_W_ENC, FSUEQ_W_DESC, Requires<[HasMSA]>;
def FSUEQ_D : FSUEQ_D_ENC, FSUEQ_D_DESC, Requires<[HasMSA]>;

def FSULE_W : FSULE_W_ENC, FSULE_W_DESC, Requires<[HasMSA]>;
def FSULE_D : FSULE_D_ENC, FSULE_D_DESC, Requires<[HasMSA]>;

def FSULT_W : FSULT_W_ENC, FSULT_W_DESC, Requires<[HasMSA]>;
def FSULT_D : FSULT_D_ENC, FSULT_D_DESC, Requires<[HasMSA]>;

def FSUN_W : FSUN_W_ENC, FSUN_W_DESC, Requires<[HasMSA]>;
def FSUN_D : FSUN_D_ENC, FSUN_D_DESC, Requires<[HasMSA]>;

def FSUNE_W : FSUNE_W_ENC, FSUNE_W_DESC, Requires<[HasMSA]>;
def FSUNE_D : FSUNE_D_ENC, FSUNE_D_DESC, Requires<[HasMSA]>;

def FTRUNC_S_W : FTRUNC_S_W_ENC, FTRUNC_S_W_DESC, Requires<[HasMSA]>;
def FTRUNC_S_D : FTRUNC_S_D_ENC, FTRUNC_S_D_DESC, Requires<[HasMSA]>;

def FTRUNC_U_W : FTRUNC_U_W_ENC, FTRUNC_U_W_DESC, Requires<[HasMSA]>;
def FTRUNC_U_D : FTRUNC_U_D_ENC, FTRUNC_U_D_DESC, Requires<[HasMSA]>;

def FTINT_S_W : FTINT_S_W_ENC, FTINT_S_W_DESC, Requires<[HasMSA]>;
def FTINT_S_D : FTINT_S_D_ENC, FTINT_S_D_DESC, Requires<[HasMSA]>;

def FTINT_U_W : FTINT_U_W_ENC, FTINT_U_W_DESC, Requires<[HasMSA]>;
def FTINT_U_D : FTINT_U_D_ENC, FTINT_U_D_DESC, Requires<[HasMSA]>;

def FTQ_H : FTQ_H_ENC, FTQ_H_DESC, Requires<[HasMSA]>;
def FTQ_W : FTQ_W_ENC, FTQ_W_DESC, Requires<[HasMSA]>;

def HADD_S_H : HADD_S_H_ENC, HADD_S_H_DESC, Requires<[HasMSA]>;
def HADD_S_W : HADD_S_W_ENC, HADD_S_W_DESC, Requires<[HasMSA]>;
def HADD_S_D : HADD_S_D_ENC, HADD_S_D_DESC, Requires<[HasMSA]>;

def HADD_U_H : HADD_U_H_ENC, HADD_U_H_DESC, Requires<[HasMSA]>;
def HADD_U_W : HADD_U_W_ENC, HADD_U_W_DESC, Requires<[HasMSA]>;
def HADD_U_D : HADD_U_D_ENC, HADD_U_D_DESC, Requires<[HasMSA]>;

def HSUB_S_H : HSUB_S_H_ENC, HSUB_S_H_DESC, Requires<[HasMSA]>;
def HSUB_S_W : HSUB_S_W_ENC, HSUB_S_W_DESC, Requires<[HasMSA]>;
def HSUB_S_D : HSUB_S_D_ENC, HSUB_S_D_DESC, Requires<[HasMSA]>;

def HSUB_U_H : HSUB_U_H_ENC, HSUB_U_H_DESC, Requires<[HasMSA]>;
def HSUB_U_W : HSUB_U_W_ENC, HSUB_U_W_DESC, Requires<[HasMSA]>;
def HSUB_U_D : HSUB_U_D_ENC, HSUB_U_D_DESC, Requires<[HasMSA]>;

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

def INSVE_B : INSVE_B_ENC, INSVE_B_DESC, Requires<[HasMSA]>;
def INSVE_H : INSVE_H_ENC, INSVE_H_DESC, Requires<[HasMSA]>;
def INSVE_W : INSVE_W_ENC, INSVE_W_DESC, Requires<[HasMSA]>;
def INSVE_D : INSVE_D_ENC, INSVE_D_DESC, Requires<[HasMSA]>;

def LD_B: LD_B_ENC, LD_B_DESC, Requires<[HasMSA]>;
def LD_H: LD_H_ENC, LD_H_DESC, Requires<[HasMSA]>;
def LD_W: LD_W_ENC, LD_W_DESC, Requires<[HasMSA]>;
def LD_D: LD_D_ENC, LD_D_DESC, Requires<[HasMSA]>;

def LDI_B : LDI_B_ENC, LDI_B_DESC, Requires<[HasMSA]>;
def LDI_H : LDI_H_ENC, LDI_H_DESC, Requires<[HasMSA]>;
def LDI_W : LDI_W_ENC, LDI_W_DESC, Requires<[HasMSA]>;

def LDX_B: LDX_B_ENC, LDX_B_DESC, Requires<[HasMSA]>;
def LDX_H: LDX_H_ENC, LDX_H_DESC, Requires<[HasMSA]>;
def LDX_W: LDX_W_ENC, LDX_W_DESC, Requires<[HasMSA]>;
def LDX_D: LDX_D_ENC, LDX_D_DESC, Requires<[HasMSA]>;

def MADD_Q_H : MADD_Q_H_ENC, MADD_Q_H_DESC, Requires<[HasMSA]>;
def MADD_Q_W : MADD_Q_W_ENC, MADD_Q_W_DESC, Requires<[HasMSA]>;

def MADDR_Q_H : MADDR_Q_H_ENC, MADDR_Q_H_DESC, Requires<[HasMSA]>;
def MADDR_Q_W : MADDR_Q_W_ENC, MADDR_Q_W_DESC, Requires<[HasMSA]>;

def MADDV_B : MADDV_B_ENC, MADDV_B_DESC, Requires<[HasMSA]>;
def MADDV_H : MADDV_H_ENC, MADDV_H_DESC, Requires<[HasMSA]>;
def MADDV_W : MADDV_W_ENC, MADDV_W_DESC, Requires<[HasMSA]>;
def MADDV_D : MADDV_D_ENC, MADDV_D_DESC, Requires<[HasMSA]>;

def MAX_A_B : MAX_A_B_ENC, MAX_A_B_DESC, Requires<[HasMSA]>;
def MAX_A_H : MAX_A_H_ENC, MAX_A_H_DESC, Requires<[HasMSA]>;
def MAX_A_W : MAX_A_W_ENC, MAX_A_W_DESC, Requires<[HasMSA]>;
def MAX_A_D : MAX_A_D_ENC, MAX_A_D_DESC, Requires<[HasMSA]>;

def MAX_S_B : MAX_S_B_ENC, MAX_S_B_DESC, Requires<[HasMSA]>;
def MAX_S_H : MAX_S_H_ENC, MAX_S_H_DESC, Requires<[HasMSA]>;
def MAX_S_W : MAX_S_W_ENC, MAX_S_W_DESC, Requires<[HasMSA]>;
def MAX_S_D : MAX_S_D_ENC, MAX_S_D_DESC, Requires<[HasMSA]>;

def MAX_U_B : MAX_U_B_ENC, MAX_U_B_DESC, Requires<[HasMSA]>;
def MAX_U_H : MAX_U_H_ENC, MAX_U_H_DESC, Requires<[HasMSA]>;
def MAX_U_W : MAX_U_W_ENC, MAX_U_W_DESC, Requires<[HasMSA]>;
def MAX_U_D : MAX_U_D_ENC, MAX_U_D_DESC, Requires<[HasMSA]>;

def MAXI_S_B : MAXI_S_B_ENC, MAXI_S_B_DESC, Requires<[HasMSA]>;
def MAXI_S_H : MAXI_S_H_ENC, MAXI_S_H_DESC, Requires<[HasMSA]>;
def MAXI_S_W : MAXI_S_W_ENC, MAXI_S_W_DESC, Requires<[HasMSA]>;
def MAXI_S_D : MAXI_S_D_ENC, MAXI_S_D_DESC, Requires<[HasMSA]>;

def MAXI_U_B : MAXI_U_B_ENC, MAXI_U_B_DESC, Requires<[HasMSA]>;
def MAXI_U_H : MAXI_U_H_ENC, MAXI_U_H_DESC, Requires<[HasMSA]>;
def MAXI_U_W : MAXI_U_W_ENC, MAXI_U_W_DESC, Requires<[HasMSA]>;
def MAXI_U_D : MAXI_U_D_ENC, MAXI_U_D_DESC, Requires<[HasMSA]>;

def MIN_A_B : MIN_A_B_ENC, MIN_A_B_DESC, Requires<[HasMSA]>;
def MIN_A_H : MIN_A_H_ENC, MIN_A_H_DESC, Requires<[HasMSA]>;
def MIN_A_W : MIN_A_W_ENC, MIN_A_W_DESC, Requires<[HasMSA]>;
def MIN_A_D : MIN_A_D_ENC, MIN_A_D_DESC, Requires<[HasMSA]>;

def MIN_S_B : MIN_S_B_ENC, MIN_S_B_DESC, Requires<[HasMSA]>;
def MIN_S_H : MIN_S_H_ENC, MIN_S_H_DESC, Requires<[HasMSA]>;
def MIN_S_W : MIN_S_W_ENC, MIN_S_W_DESC, Requires<[HasMSA]>;
def MIN_S_D : MIN_S_D_ENC, MIN_S_D_DESC, Requires<[HasMSA]>;

def MIN_U_B : MIN_U_B_ENC, MIN_U_B_DESC, Requires<[HasMSA]>;
def MIN_U_H : MIN_U_H_ENC, MIN_U_H_DESC, Requires<[HasMSA]>;
def MIN_U_W : MIN_U_W_ENC, MIN_U_W_DESC, Requires<[HasMSA]>;
def MIN_U_D : MIN_U_D_ENC, MIN_U_D_DESC, Requires<[HasMSA]>;

def MINI_S_B : MINI_S_B_ENC, MINI_S_B_DESC, Requires<[HasMSA]>;
def MINI_S_H : MINI_S_H_ENC, MINI_S_H_DESC, Requires<[HasMSA]>;
def MINI_S_W : MINI_S_W_ENC, MINI_S_W_DESC, Requires<[HasMSA]>;
def MINI_S_D : MINI_S_D_ENC, MINI_S_D_DESC, Requires<[HasMSA]>;

def MINI_U_B : MINI_U_B_ENC, MINI_U_B_DESC, Requires<[HasMSA]>;
def MINI_U_H : MINI_U_H_ENC, MINI_U_H_DESC, Requires<[HasMSA]>;
def MINI_U_W : MINI_U_W_ENC, MINI_U_W_DESC, Requires<[HasMSA]>;
def MINI_U_D : MINI_U_D_ENC, MINI_U_D_DESC, Requires<[HasMSA]>;

def MOD_S_B : MOD_S_B_ENC, MOD_S_B_DESC, Requires<[HasMSA]>;
def MOD_S_H : MOD_S_H_ENC, MOD_S_H_DESC, Requires<[HasMSA]>;
def MOD_S_W : MOD_S_W_ENC, MOD_S_W_DESC, Requires<[HasMSA]>;
def MOD_S_D : MOD_S_D_ENC, MOD_S_D_DESC, Requires<[HasMSA]>;

def MOD_U_B : MOD_U_B_ENC, MOD_U_B_DESC, Requires<[HasMSA]>;
def MOD_U_H : MOD_U_H_ENC, MOD_U_H_DESC, Requires<[HasMSA]>;
def MOD_U_W : MOD_U_W_ENC, MOD_U_W_DESC, Requires<[HasMSA]>;
def MOD_U_D : MOD_U_D_ENC, MOD_U_D_DESC, Requires<[HasMSA]>;

def MOVE_V : MOVE_V_ENC, MOVE_V_DESC, Requires<[HasMSA]>;

def MSUB_Q_H : MSUB_Q_H_ENC, MSUB_Q_H_DESC, Requires<[HasMSA]>;
def MSUB_Q_W : MSUB_Q_W_ENC, MSUB_Q_W_DESC, Requires<[HasMSA]>;

def MSUBR_Q_H : MSUBR_Q_H_ENC, MSUBR_Q_H_DESC, Requires<[HasMSA]>;
def MSUBR_Q_W : MSUBR_Q_W_ENC, MSUBR_Q_W_DESC, Requires<[HasMSA]>;

def MSUBV_B : MSUBV_B_ENC, MSUBV_B_DESC, Requires<[HasMSA]>;
def MSUBV_H : MSUBV_H_ENC, MSUBV_H_DESC, Requires<[HasMSA]>;
def MSUBV_W : MSUBV_W_ENC, MSUBV_W_DESC, Requires<[HasMSA]>;
def MSUBV_D : MSUBV_D_ENC, MSUBV_D_DESC, Requires<[HasMSA]>;

def MUL_Q_H : MUL_Q_H_ENC, MUL_Q_H_DESC, Requires<[HasMSA]>;
def MUL_Q_W : MUL_Q_W_ENC, MUL_Q_W_DESC, Requires<[HasMSA]>;

def MULR_Q_H : MULR_Q_H_ENC, MULR_Q_H_DESC, Requires<[HasMSA]>;
def MULR_Q_W : MULR_Q_W_ENC, MULR_Q_W_DESC, Requires<[HasMSA]>;

def MULV_B : MULV_B_ENC, MULV_B_DESC, Requires<[HasMSA]>;
def MULV_H : MULV_H_ENC, MULV_H_DESC, Requires<[HasMSA]>;
def MULV_W : MULV_W_ENC, MULV_W_DESC, Requires<[HasMSA]>;
def MULV_D : MULV_D_ENC, MULV_D_DESC, Requires<[HasMSA]>;

def NLOC_B : NLOC_B_ENC, NLOC_B_DESC, Requires<[HasMSA]>;
def NLOC_H : NLOC_H_ENC, NLOC_H_DESC, Requires<[HasMSA]>;
def NLOC_W : NLOC_W_ENC, NLOC_W_DESC, Requires<[HasMSA]>;
def NLOC_D : NLOC_D_ENC, NLOC_D_DESC, Requires<[HasMSA]>;

def NLZC_B : NLZC_B_ENC, NLZC_B_DESC, Requires<[HasMSA]>;
def NLZC_H : NLZC_H_ENC, NLZC_H_DESC, Requires<[HasMSA]>;
def NLZC_W : NLZC_W_ENC, NLZC_W_DESC, Requires<[HasMSA]>;
def NLZC_D : NLZC_D_ENC, NLZC_D_DESC, Requires<[HasMSA]>;

def NOR_V : NOR_V_ENC, NOR_V_DESC, Requires<[HasMSA]>;

def NORI_B : NORI_B_ENC, NORI_B_DESC, Requires<[HasMSA]>;

def OR_V : OR_V_ENC, OR_V_DESC, Requires<[HasMSA]>;

def ORI_B : ORI_B_ENC, ORI_B_DESC, Requires<[HasMSA]>;

def PCKEV_B : PCKEV_B_ENC, PCKEV_B_DESC, Requires<[HasMSA]>;
def PCKEV_H : PCKEV_H_ENC, PCKEV_H_DESC, Requires<[HasMSA]>;
def PCKEV_W : PCKEV_W_ENC, PCKEV_W_DESC, Requires<[HasMSA]>;
def PCKEV_D : PCKEV_D_ENC, PCKEV_D_DESC, Requires<[HasMSA]>;

def PCKOD_B : PCKOD_B_ENC, PCKOD_B_DESC, Requires<[HasMSA]>;
def PCKOD_H : PCKOD_H_ENC, PCKOD_H_DESC, Requires<[HasMSA]>;
def PCKOD_W : PCKOD_W_ENC, PCKOD_W_DESC, Requires<[HasMSA]>;
def PCKOD_D : PCKOD_D_ENC, PCKOD_D_DESC, Requires<[HasMSA]>;

def PCNT_B : PCNT_B_ENC, PCNT_B_DESC, Requires<[HasMSA]>;
def PCNT_H : PCNT_H_ENC, PCNT_H_DESC, Requires<[HasMSA]>;
def PCNT_W : PCNT_W_ENC, PCNT_W_DESC, Requires<[HasMSA]>;
def PCNT_D : PCNT_D_ENC, PCNT_D_DESC, Requires<[HasMSA]>;

def SAT_S_B : SAT_S_B_ENC, SAT_S_B_DESC, Requires<[HasMSA]>;
def SAT_S_H : SAT_S_H_ENC, SAT_S_H_DESC, Requires<[HasMSA]>;
def SAT_S_W : SAT_S_W_ENC, SAT_S_W_DESC, Requires<[HasMSA]>;
def SAT_S_D : SAT_S_D_ENC, SAT_S_D_DESC, Requires<[HasMSA]>;

def SAT_U_B : SAT_U_B_ENC, SAT_U_B_DESC, Requires<[HasMSA]>;
def SAT_U_H : SAT_U_H_ENC, SAT_U_H_DESC, Requires<[HasMSA]>;
def SAT_U_W : SAT_U_W_ENC, SAT_U_W_DESC, Requires<[HasMSA]>;
def SAT_U_D : SAT_U_D_ENC, SAT_U_D_DESC, Requires<[HasMSA]>;

def SHF_B : SHF_B_ENC, SHF_B_DESC, Requires<[HasMSA]>;
def SHF_H : SHF_H_ENC, SHF_H_DESC, Requires<[HasMSA]>;
def SHF_W : SHF_W_ENC, SHF_W_DESC, Requires<[HasMSA]>;

def SLD_B : SLD_B_ENC, SLD_B_DESC, Requires<[HasMSA]>;
def SLD_H : SLD_H_ENC, SLD_H_DESC, Requires<[HasMSA]>;
def SLD_W : SLD_W_ENC, SLD_W_DESC, Requires<[HasMSA]>;
def SLD_D : SLD_D_ENC, SLD_D_DESC, Requires<[HasMSA]>;

def SLDI_B : SLDI_B_ENC, SLDI_B_DESC, Requires<[HasMSA]>;
def SLDI_H : SLDI_H_ENC, SLDI_H_DESC, Requires<[HasMSA]>;
def SLDI_W : SLDI_W_ENC, SLDI_W_DESC, Requires<[HasMSA]>;
def SLDI_D : SLDI_D_ENC, SLDI_D_DESC, Requires<[HasMSA]>;

def SLL_B : SLL_B_ENC, SLL_B_DESC, Requires<[HasMSA]>;
def SLL_H : SLL_H_ENC, SLL_H_DESC, Requires<[HasMSA]>;
def SLL_W : SLL_W_ENC, SLL_W_DESC, Requires<[HasMSA]>;
def SLL_D : SLL_D_ENC, SLL_D_DESC, Requires<[HasMSA]>;

def SLLI_B : SLLI_B_ENC, SLLI_B_DESC, Requires<[HasMSA]>;
def SLLI_H : SLLI_H_ENC, SLLI_H_DESC, Requires<[HasMSA]>;
def SLLI_W : SLLI_W_ENC, SLLI_W_DESC, Requires<[HasMSA]>;
def SLLI_D : SLLI_D_ENC, SLLI_D_DESC, Requires<[HasMSA]>;

def SPLAT_B : SPLAT_B_ENC, SPLAT_B_DESC, Requires<[HasMSA]>;
def SPLAT_H : SPLAT_H_ENC, SPLAT_H_DESC, Requires<[HasMSA]>;
def SPLAT_W : SPLAT_W_ENC, SPLAT_W_DESC, Requires<[HasMSA]>;
def SPLAT_D : SPLAT_D_ENC, SPLAT_D_DESC, Requires<[HasMSA]>;

def SPLATI_B : SPLATI_B_ENC, SPLATI_B_DESC, Requires<[HasMSA]>;
def SPLATI_H : SPLATI_H_ENC, SPLATI_H_DESC, Requires<[HasMSA]>;
def SPLATI_W : SPLATI_W_ENC, SPLATI_W_DESC, Requires<[HasMSA]>;
def SPLATI_D : SPLATI_D_ENC, SPLATI_D_DESC, Requires<[HasMSA]>;

def SRA_B : SRA_B_ENC, SRA_B_DESC, Requires<[HasMSA]>;
def SRA_H : SRA_H_ENC, SRA_H_DESC, Requires<[HasMSA]>;
def SRA_W : SRA_W_ENC, SRA_W_DESC, Requires<[HasMSA]>;
def SRA_D : SRA_D_ENC, SRA_D_DESC, Requires<[HasMSA]>;

def SRAI_B : SRAI_B_ENC, SRAI_B_DESC, Requires<[HasMSA]>;
def SRAI_H : SRAI_H_ENC, SRAI_H_DESC, Requires<[HasMSA]>;
def SRAI_W : SRAI_W_ENC, SRAI_W_DESC, Requires<[HasMSA]>;
def SRAI_D : SRAI_D_ENC, SRAI_D_DESC, Requires<[HasMSA]>;

def SRAR_B : SRAR_B_ENC, SRAR_B_DESC, Requires<[HasMSA]>;
def SRAR_H : SRAR_H_ENC, SRAR_H_DESC, Requires<[HasMSA]>;
def SRAR_W : SRAR_W_ENC, SRAR_W_DESC, Requires<[HasMSA]>;
def SRAR_D : SRAR_D_ENC, SRAR_D_DESC, Requires<[HasMSA]>;

def SRARI_B : SRARI_B_ENC, SRARI_B_DESC, Requires<[HasMSA]>;
def SRARI_H : SRARI_H_ENC, SRARI_H_DESC, Requires<[HasMSA]>;
def SRARI_W : SRARI_W_ENC, SRARI_W_DESC, Requires<[HasMSA]>;
def SRARI_D : SRARI_D_ENC, SRARI_D_DESC, Requires<[HasMSA]>;

def SRL_B : SRL_B_ENC, SRL_B_DESC, Requires<[HasMSA]>;
def SRL_H : SRL_H_ENC, SRL_H_DESC, Requires<[HasMSA]>;
def SRL_W : SRL_W_ENC, SRL_W_DESC, Requires<[HasMSA]>;
def SRL_D : SRL_D_ENC, SRL_D_DESC, Requires<[HasMSA]>;

def SRLI_B : SRLI_B_ENC, SRLI_B_DESC, Requires<[HasMSA]>;
def SRLI_H : SRLI_H_ENC, SRLI_H_DESC, Requires<[HasMSA]>;
def SRLI_W : SRLI_W_ENC, SRLI_W_DESC, Requires<[HasMSA]>;
def SRLI_D : SRLI_D_ENC, SRLI_D_DESC, Requires<[HasMSA]>;

def SRLR_B : SRLR_B_ENC, SRLR_B_DESC, Requires<[HasMSA]>;
def SRLR_H : SRLR_H_ENC, SRLR_H_DESC, Requires<[HasMSA]>;
def SRLR_W : SRLR_W_ENC, SRLR_W_DESC, Requires<[HasMSA]>;
def SRLR_D : SRLR_D_ENC, SRLR_D_DESC, Requires<[HasMSA]>;

def SRLRI_B : SRLRI_B_ENC, SRLRI_B_DESC, Requires<[HasMSA]>;
def SRLRI_H : SRLRI_H_ENC, SRLRI_H_DESC, Requires<[HasMSA]>;
def SRLRI_W : SRLRI_W_ENC, SRLRI_W_DESC, Requires<[HasMSA]>;
def SRLRI_D : SRLRI_D_ENC, SRLRI_D_DESC, Requires<[HasMSA]>;

def ST_B: ST_B_ENC, ST_B_DESC, Requires<[HasMSA]>;
def ST_H: ST_H_ENC, ST_H_DESC, Requires<[HasMSA]>;
def ST_W: ST_W_ENC, ST_W_DESC, Requires<[HasMSA]>;
def ST_D: ST_D_ENC, ST_D_DESC, Requires<[HasMSA]>;

def STX_B: STX_B_ENC, STX_B_DESC, Requires<[HasMSA]>;
def STX_H: STX_H_ENC, STX_H_DESC, Requires<[HasMSA]>;
def STX_W: STX_W_ENC, STX_W_DESC, Requires<[HasMSA]>;
def STX_D: STX_D_ENC, STX_D_DESC, Requires<[HasMSA]>;

def SUBS_S_B : SUBS_S_B_ENC, SUBS_S_B_DESC, Requires<[HasMSA]>;
def SUBS_S_H : SUBS_S_H_ENC, SUBS_S_H_DESC, Requires<[HasMSA]>;
def SUBS_S_W : SUBS_S_W_ENC, SUBS_S_W_DESC, Requires<[HasMSA]>;
def SUBS_S_D : SUBS_S_D_ENC, SUBS_S_D_DESC, Requires<[HasMSA]>;

def SUBS_U_B : SUBS_U_B_ENC, SUBS_U_B_DESC, Requires<[HasMSA]>;
def SUBS_U_H : SUBS_U_H_ENC, SUBS_U_H_DESC, Requires<[HasMSA]>;
def SUBS_U_W : SUBS_U_W_ENC, SUBS_U_W_DESC, Requires<[HasMSA]>;
def SUBS_U_D : SUBS_U_D_ENC, SUBS_U_D_DESC, Requires<[HasMSA]>;

def SUBSUS_U_B : SUBSUS_U_B_ENC, SUBSUS_U_B_DESC, Requires<[HasMSA]>;
def SUBSUS_U_H : SUBSUS_U_H_ENC, SUBSUS_U_H_DESC, Requires<[HasMSA]>;
def SUBSUS_U_W : SUBSUS_U_W_ENC, SUBSUS_U_W_DESC, Requires<[HasMSA]>;
def SUBSUS_U_D : SUBSUS_U_D_ENC, SUBSUS_U_D_DESC, Requires<[HasMSA]>;

def SUBSUU_S_B : SUBSUU_S_B_ENC, SUBSUU_S_B_DESC, Requires<[HasMSA]>;
def SUBSUU_S_H : SUBSUU_S_H_ENC, SUBSUU_S_H_DESC, Requires<[HasMSA]>;
def SUBSUU_S_W : SUBSUU_S_W_ENC, SUBSUU_S_W_DESC, Requires<[HasMSA]>;
def SUBSUU_S_D : SUBSUU_S_D_ENC, SUBSUU_S_D_DESC, Requires<[HasMSA]>;

def SUBV_B : SUBV_B_ENC, SUBV_B_DESC, Requires<[HasMSA]>;
def SUBV_H : SUBV_H_ENC, SUBV_H_DESC, Requires<[HasMSA]>;
def SUBV_W : SUBV_W_ENC, SUBV_W_DESC, Requires<[HasMSA]>;
def SUBV_D : SUBV_D_ENC, SUBV_D_DESC, Requires<[HasMSA]>;

def SUBVI_B : SUBVI_B_ENC, SUBVI_B_DESC, Requires<[HasMSA]>;
def SUBVI_H : SUBVI_H_ENC, SUBVI_H_DESC, Requires<[HasMSA]>;
def SUBVI_W : SUBVI_W_ENC, SUBVI_W_DESC, Requires<[HasMSA]>;
def SUBVI_D : SUBVI_D_ENC, SUBVI_D_DESC, Requires<[HasMSA]>;

def VSHF_B : VSHF_B_ENC, VSHF_B_DESC, Requires<[HasMSA]>;
def VSHF_H : VSHF_H_ENC, VSHF_H_DESC, Requires<[HasMSA]>;
def VSHF_W : VSHF_W_ENC, VSHF_W_DESC, Requires<[HasMSA]>;
def VSHF_D : VSHF_D_ENC, VSHF_D_DESC, Requires<[HasMSA]>;

def XOR_V : XOR_V_ENC, XOR_V_DESC, Requires<[HasMSA]>;

def XORI_B : XORI_B_ENC, XORI_B_DESC, Requires<[HasMSA]>;

// Patterns.
class MSAPat<dag pattern, dag result, list<Predicate> pred = [HasMSA]> :
  Pat<pattern, result>, Requires<pred>;

def : MSAPat<(v16i8 (load addr:$addr)), (LD_B addr:$addr)>;
def : MSAPat<(v8i16 (load addr:$addr)), (LD_H addr:$addr)>;
def : MSAPat<(v4i32 (load addr:$addr)), (LD_W addr:$addr)>;
def : MSAPat<(v2i64 (load addr:$addr)), (LD_D addr:$addr)>;
def : MSAPat<(v8f16 (load addr:$addr)), (LD_H addr:$addr)>;
def : MSAPat<(v4f32 (load addr:$addr)), (LD_W addr:$addr)>;
def : MSAPat<(v2f64 (load addr:$addr)), (LD_D addr:$addr)>;

def : MSAPat<(v8f16 (load addrRegImm:$addr)), (LD_H addrRegImm:$addr)>;
def : MSAPat<(v4f32 (load addrRegImm:$addr)), (LD_W addrRegImm:$addr)>;
def : MSAPat<(v2f64 (load addrRegImm:$addr)), (LD_D addrRegImm:$addr)>;

def : MSAPat<(store (v16i8 MSA128B:$ws), addr:$addr),
             (ST_B MSA128B:$ws, addr:$addr)>;
def : MSAPat<(store (v8i16 MSA128H:$ws), addr:$addr),
             (ST_H MSA128H:$ws, addr:$addr)>;
def : MSAPat<(store (v4i32 MSA128W:$ws), addr:$addr),
             (ST_W MSA128W:$ws, addr:$addr)>;
def : MSAPat<(store (v2i64 MSA128D:$ws), addr:$addr),
             (ST_D MSA128D:$ws, addr:$addr)>;
def : MSAPat<(store (v8f16 MSA128H:$ws), addr:$addr),
             (ST_H MSA128H:$ws, addr:$addr)>;
def : MSAPat<(store (v4f32 MSA128W:$ws), addr:$addr),
             (ST_W MSA128W:$ws, addr:$addr)>;
def : MSAPat<(store (v2f64 MSA128D:$ws), addr:$addr),
             (ST_D MSA128D:$ws, addr:$addr)>;

def ST_FH : MSAPat<(store (v8f16 MSA128H:$ws), addrRegImm:$addr),
                   (ST_H MSA128H:$ws, addrRegImm:$addr)>;
def ST_FW : MSAPat<(store (v4f32 MSA128W:$ws), addrRegImm:$addr),
                   (ST_W MSA128W:$ws, addrRegImm:$addr)>;
def ST_FD : MSAPat<(store (v2f64 MSA128D:$ws), addrRegImm:$addr),
                   (ST_D MSA128D:$ws, addrRegImm:$addr)>;

class MSABitconvertPat<ValueType DstVT, ValueType SrcVT,
                       RegisterClass DstRC, list<Predicate> preds = [HasMSA]> :
   MSAPat<(DstVT (bitconvert SrcVT:$src)),
          (COPY_TO_REGCLASS SrcVT:$src, DstRC), preds>;

// These are endian-independant because the element size doesnt change
def : MSABitconvertPat<v8i16, v8f16, MSA128H>;
def : MSABitconvertPat<v4i32, v4f32, MSA128W>;
def : MSABitconvertPat<v2i64, v2f64, MSA128D>;
def : MSABitconvertPat<v8f16, v8i16, MSA128H>;
def : MSABitconvertPat<v4f32, v4i32, MSA128W>;
def : MSABitconvertPat<v2f64, v2i64, MSA128D>;

// Little endian bitcasts are always no-ops
def : MSABitconvertPat<v16i8, v8i16, MSA128B, [HasMSA, IsLE]>;
def : MSABitconvertPat<v16i8, v4i32, MSA128B, [HasMSA, IsLE]>;
def : MSABitconvertPat<v16i8, v2i64, MSA128B, [HasMSA, IsLE]>;
def : MSABitconvertPat<v16i8, v8f16, MSA128B, [HasMSA, IsLE]>;
def : MSABitconvertPat<v16i8, v4f32, MSA128B, [HasMSA, IsLE]>;
def : MSABitconvertPat<v16i8, v2f64, MSA128B, [HasMSA, IsLE]>;

def : MSABitconvertPat<v8i16, v16i8, MSA128H, [HasMSA, IsLE]>;
def : MSABitconvertPat<v8i16, v4i32, MSA128H, [HasMSA, IsLE]>;
def : MSABitconvertPat<v8i16, v2i64, MSA128H, [HasMSA, IsLE]>;
def : MSABitconvertPat<v8i16, v4f32, MSA128H, [HasMSA, IsLE]>;
def : MSABitconvertPat<v8i16, v2f64, MSA128H, [HasMSA, IsLE]>;

def : MSABitconvertPat<v4i32, v16i8, MSA128W, [HasMSA, IsLE]>;
def : MSABitconvertPat<v4i32, v8i16, MSA128W, [HasMSA, IsLE]>;
def : MSABitconvertPat<v4i32, v2i64, MSA128W, [HasMSA, IsLE]>;
def : MSABitconvertPat<v4i32, v8f16, MSA128W, [HasMSA, IsLE]>;
def : MSABitconvertPat<v4i32, v2f64, MSA128W, [HasMSA, IsLE]>;

def : MSABitconvertPat<v2i64, v16i8, MSA128D, [HasMSA, IsLE]>;
def : MSABitconvertPat<v2i64, v8i16, MSA128D, [HasMSA, IsLE]>;
def : MSABitconvertPat<v2i64, v4i32, MSA128D, [HasMSA, IsLE]>;
def : MSABitconvertPat<v2i64, v8f16, MSA128D, [HasMSA, IsLE]>;
def : MSABitconvertPat<v2i64, v4f32, MSA128D, [HasMSA, IsLE]>;

def : MSABitconvertPat<v4f32, v16i8, MSA128W, [HasMSA, IsLE]>;
def : MSABitconvertPat<v4f32, v8i16, MSA128W, [HasMSA, IsLE]>;
def : MSABitconvertPat<v4f32, v2i64, MSA128W, [HasMSA, IsLE]>;
def : MSABitconvertPat<v4f32, v8f16, MSA128W, [HasMSA, IsLE]>;
def : MSABitconvertPat<v4f32, v2f64, MSA128W, [HasMSA, IsLE]>;

def : MSABitconvertPat<v2f64, v16i8, MSA128D, [HasMSA, IsLE]>;
def : MSABitconvertPat<v2f64, v8i16, MSA128D, [HasMSA, IsLE]>;
def : MSABitconvertPat<v2f64, v4i32, MSA128D, [HasMSA, IsLE]>;
def : MSABitconvertPat<v2f64, v8f16, MSA128D, [HasMSA, IsLE]>;
def : MSABitconvertPat<v2f64, v4f32, MSA128D, [HasMSA, IsLE]>;

// Big endian bitcasts expand to shuffle instructions.
// This is because bitcast is defined to be a store/load sequence and the
// vector store/load instructions are mixed-endian with respect to the vector
// as a whole (little endian with respect to element order, but big endian
// elements).

class MSABitconvertReverseQuartersPat<ValueType DstVT, ValueType SrcVT,
                                      RegisterClass DstRC, MSAInst Insn,
                                      RegisterClass ViaRC> :
  MSAPat<(DstVT (bitconvert SrcVT:$src)),
         (COPY_TO_REGCLASS (Insn (COPY_TO_REGCLASS SrcVT:$src, ViaRC), 27),
                           DstRC),
         [HasMSA, IsBE]>;

class MSABitconvertReverseHalvesPat<ValueType DstVT, ValueType SrcVT,
                                    RegisterClass DstRC, MSAInst Insn,
                                    RegisterClass ViaRC> :
  MSAPat<(DstVT (bitconvert SrcVT:$src)),
         (COPY_TO_REGCLASS (Insn (COPY_TO_REGCLASS SrcVT:$src, ViaRC), 177),
                           DstRC),
         [HasMSA, IsBE]>;

class MSABitconvertReverseBInHPat<ValueType DstVT, ValueType SrcVT,
                                  RegisterClass DstRC> :
  MSABitconvertReverseHalvesPat<DstVT, SrcVT, DstRC, SHF_B, MSA128B>;

class MSABitconvertReverseBInWPat<ValueType DstVT, ValueType SrcVT,
                                  RegisterClass DstRC> :
  MSABitconvertReverseQuartersPat<DstVT, SrcVT, DstRC, SHF_B, MSA128B>;

class MSABitconvertReverseBInDPat<ValueType DstVT, ValueType SrcVT,
                                  RegisterClass DstRC> :
  MSAPat<(DstVT (bitconvert SrcVT:$src)),
         (COPY_TO_REGCLASS
           (SHF_W
             (COPY_TO_REGCLASS
               (SHF_B (COPY_TO_REGCLASS SrcVT:$src, MSA128B), 27),
               MSA128W), 177),
           DstRC),
         [HasMSA, IsBE]>;

class MSABitconvertReverseHInWPat<ValueType DstVT, ValueType SrcVT,
                                  RegisterClass DstRC> :
  MSABitconvertReverseHalvesPat<DstVT, SrcVT, DstRC, SHF_H, MSA128H>;

class MSABitconvertReverseHInDPat<ValueType DstVT, ValueType SrcVT,
                                  RegisterClass DstRC> :
  MSABitconvertReverseQuartersPat<DstVT, SrcVT, DstRC, SHF_H, MSA128H>;

class MSABitconvertReverseWInDPat<ValueType DstVT, ValueType SrcVT,
                                  RegisterClass DstRC> :
  MSABitconvertReverseHalvesPat<DstVT, SrcVT, DstRC, SHF_W, MSA128W>;

def : MSABitconvertReverseBInHPat<v8i16, v16i8, MSA128H>;
def : MSABitconvertReverseBInHPat<v8f16, v16i8, MSA128H>;
def : MSABitconvertReverseBInWPat<v4i32, v16i8, MSA128W>;
def : MSABitconvertReverseBInWPat<v4f32, v16i8, MSA128W>;
def : MSABitconvertReverseBInDPat<v2i64, v16i8, MSA128D>;
def : MSABitconvertReverseBInDPat<v2f64, v16i8, MSA128D>;

def : MSABitconvertReverseBInHPat<v16i8, v8i16, MSA128B>;
def : MSABitconvertReverseHInWPat<v4i32, v8i16, MSA128W>;
def : MSABitconvertReverseHInWPat<v4f32, v8i16, MSA128W>;
def : MSABitconvertReverseHInDPat<v2i64, v8i16, MSA128D>;
def : MSABitconvertReverseHInDPat<v2f64, v8i16, MSA128D>;

def : MSABitconvertReverseBInHPat<v16i8, v8f16, MSA128B>;
def : MSABitconvertReverseHInWPat<v4i32, v8f16, MSA128W>;
def : MSABitconvertReverseHInWPat<v4f32, v8f16, MSA128W>;
def : MSABitconvertReverseHInDPat<v2i64, v8f16, MSA128D>;
def : MSABitconvertReverseHInDPat<v2f64, v8f16, MSA128D>;

def : MSABitconvertReverseBInWPat<v16i8, v4i32, MSA128B>;
def : MSABitconvertReverseHInWPat<v8i16, v4i32, MSA128H>;
def : MSABitconvertReverseHInWPat<v8f16, v4i32, MSA128H>;
def : MSABitconvertReverseWInDPat<v2i64, v4i32, MSA128D>;
def : MSABitconvertReverseWInDPat<v2f64, v4i32, MSA128D>;

def : MSABitconvertReverseBInWPat<v16i8, v4f32, MSA128B>;
def : MSABitconvertReverseHInWPat<v8i16, v4f32, MSA128H>;
def : MSABitconvertReverseHInWPat<v8f16, v4f32, MSA128H>;
def : MSABitconvertReverseWInDPat<v2i64, v4f32, MSA128D>;
def : MSABitconvertReverseWInDPat<v2f64, v4f32, MSA128D>;

def : MSABitconvertReverseBInDPat<v16i8, v2i64, MSA128B>;
def : MSABitconvertReverseHInDPat<v8i16, v2i64, MSA128H>;
def : MSABitconvertReverseHInDPat<v8f16, v2i64, MSA128H>;
def : MSABitconvertReverseWInDPat<v4i32, v2i64, MSA128W>;
def : MSABitconvertReverseWInDPat<v4f32, v2i64, MSA128W>;

def : MSABitconvertReverseBInDPat<v16i8, v2f64, MSA128B>;
def : MSABitconvertReverseHInDPat<v8i16, v2f64, MSA128H>;
def : MSABitconvertReverseHInDPat<v8f16, v2f64, MSA128H>;
def : MSABitconvertReverseWInDPat<v4i32, v2f64, MSA128W>;
def : MSABitconvertReverseWInDPat<v4f32, v2f64, MSA128W>;

// Pseudos used to implement BNZ.df, and BZ.df

class MSA_CBRANCH_PSEUDO_DESC_BASE<SDPatternOperator OpNode, ValueType TyNode,
                                   RegisterClass RCWS, InstrItinClass itin> :
  MipsPseudo<(outs GPR32:$dst),
             (ins RCWS:$ws),
             [(set GPR32:$dst, (OpNode (TyNode RCWS:$ws)))]> {
  bit usesCustomInserter = 1;
}

def SNZ_B_PSEUDO : MSA_CBRANCH_PSEUDO_DESC_BASE<MipsVAllNonZero, v16i8,
                                                MSA128B, NoItinerary>;
def SNZ_H_PSEUDO : MSA_CBRANCH_PSEUDO_DESC_BASE<MipsVAllNonZero, v8i16,
                                                MSA128H, NoItinerary>;
def SNZ_W_PSEUDO : MSA_CBRANCH_PSEUDO_DESC_BASE<MipsVAllNonZero, v4i32,
                                                MSA128W, NoItinerary>;
def SNZ_D_PSEUDO : MSA_CBRANCH_PSEUDO_DESC_BASE<MipsVAllNonZero, v2i64,
                                                MSA128D, NoItinerary>;
def SNZ_V_PSEUDO : MSA_CBRANCH_PSEUDO_DESC_BASE<MipsVAnyNonZero, v16i8,
                                                MSA128B, NoItinerary>;

def SZ_B_PSEUDO : MSA_CBRANCH_PSEUDO_DESC_BASE<MipsVAllZero, v16i8,
                                               MSA128B, NoItinerary>;
def SZ_H_PSEUDO : MSA_CBRANCH_PSEUDO_DESC_BASE<MipsVAllZero, v8i16,
                                               MSA128H, NoItinerary>;
def SZ_W_PSEUDO : MSA_CBRANCH_PSEUDO_DESC_BASE<MipsVAllZero, v4i32,
                                               MSA128W, NoItinerary>;
def SZ_D_PSEUDO : MSA_CBRANCH_PSEUDO_DESC_BASE<MipsVAllZero, v2i64,
                                               MSA128D, NoItinerary>;
def SZ_V_PSEUDO : MSA_CBRANCH_PSEUDO_DESC_BASE<MipsVAnyZero, v16i8,
                                               MSA128B, NoItinerary>;
