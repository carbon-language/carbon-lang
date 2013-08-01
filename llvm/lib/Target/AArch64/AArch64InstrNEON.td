//===-- AArch64InstrNEON.td - NEON support for AArch64 -----*- tablegen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file describes the AArch64 NEON instruction set.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// NEON-specific DAG Nodes.
//===----------------------------------------------------------------------===//
def Neon_bsl       : SDNode<"AArch64ISD::NEON_BSL", SDTypeProfile<1, 3,
                      [SDTCisVec<0>, SDTCisSameAs<0, 1>, SDTCisSameAs<0, 2>,
                      SDTCisSameAs<0, 3>]>>;

// (outs Result), (ins Imm, OpCmode)
def SDT_Neon_movi : SDTypeProfile<1, 2, [SDTCisVec<0>, SDTCisVT<1, i32>]>;

def Neon_movi     : SDNode<"AArch64ISD::NEON_MOVIMM", SDT_Neon_movi>;

def Neon_mvni     : SDNode<"AArch64ISD::NEON_MVNIMM", SDT_Neon_movi>;

// (outs Result), (ins Imm)
def Neon_fmovi : SDNode<"AArch64ISD::NEON_FMOVIMM", SDTypeProfile<1, 1,
                        [SDTCisVec<0>, SDTCisVT<1, i32>]>>;

// (outs Result), (ins LHS, RHS, CondCode)
def Neon_cmp : SDNode<"AArch64ISD::NEON_CMP", SDTypeProfile<1, 3,
                 [SDTCisVec<0>,  SDTCisSameAs<1, 2>]>>;

// (outs Result), (ins LHS, 0/0.0 constant, CondCode)
def Neon_cmpz : SDNode<"AArch64ISD::NEON_CMPZ", SDTypeProfile<1, 3,
                 [SDTCisVec<0>,  SDTCisVec<1>]>>;

// (outs Result), (ins LHS, RHS)
def Neon_tst : SDNode<"AArch64ISD::NEON_TST", SDTypeProfile<1, 2,
                 [SDTCisVec<0>,  SDTCisSameAs<1, 2>]>>;

//===----------------------------------------------------------------------===//
// Multiclasses
//===----------------------------------------------------------------------===//

multiclass NeonI_3VSame_B_sizes<bit u, bits<2> size,  bits<5> opcode,
                                string asmop, SDPatternOperator opnode8B,
                                SDPatternOperator opnode16B,
                                bit Commutable = 0>
{
  let isCommutable = Commutable in {
    def _8B :  NeonI_3VSame<0b0, u, size, opcode,
               (outs VPR64:$Rd), (ins VPR64:$Rn, VPR64:$Rm),
               asmop # "\t$Rd.8b, $Rn.8b, $Rm.8b",
               [(set (v8i8 VPR64:$Rd),
                  (v8i8 (opnode8B (v8i8 VPR64:$Rn), (v8i8 VPR64:$Rm))))],
               NoItinerary>;

    def _16B : NeonI_3VSame<0b1, u, size, opcode,
               (outs VPR128:$Rd), (ins VPR128:$Rn, VPR128:$Rm),
               asmop # "\t$Rd.16b, $Rn.16b, $Rm.16b",
               [(set (v16i8 VPR128:$Rd),
                  (v16i8 (opnode16B (v16i8 VPR128:$Rn), (v16i8 VPR128:$Rm))))],
               NoItinerary>;
  }

}

multiclass NeonI_3VSame_HS_sizes<bit u, bits<5> opcode,
                                  string asmop, SDPatternOperator opnode,
                                  bit Commutable = 0>
{
  let isCommutable = Commutable in {
    def _4H : NeonI_3VSame<0b0, u, 0b01, opcode,
              (outs VPR64:$Rd), (ins VPR64:$Rn, VPR64:$Rm),
              asmop # "\t$Rd.4h, $Rn.4h, $Rm.4h",
              [(set (v4i16 VPR64:$Rd),
                 (v4i16 (opnode (v4i16 VPR64:$Rn), (v4i16 VPR64:$Rm))))],
              NoItinerary>;

    def _8H : NeonI_3VSame<0b1, u, 0b01, opcode,
              (outs VPR128:$Rd), (ins VPR128:$Rn, VPR128:$Rm),
              asmop # "\t$Rd.8h, $Rn.8h, $Rm.8h",
              [(set (v8i16 VPR128:$Rd),
                 (v8i16 (opnode (v8i16 VPR128:$Rn), (v8i16 VPR128:$Rm))))],
              NoItinerary>;

    def _2S : NeonI_3VSame<0b0, u, 0b10, opcode,
              (outs VPR64:$Rd), (ins VPR64:$Rn, VPR64:$Rm),
              asmop # "\t$Rd.2s, $Rn.2s, $Rm.2s",
              [(set (v2i32 VPR64:$Rd),
                 (v2i32 (opnode (v2i32 VPR64:$Rn), (v2i32 VPR64:$Rm))))],
              NoItinerary>;

    def _4S : NeonI_3VSame<0b1, u, 0b10, opcode,
              (outs VPR128:$Rd), (ins VPR128:$Rn, VPR128:$Rm),
              asmop # "\t$Rd.4s, $Rn.4s, $Rm.4s",
              [(set (v4i32 VPR128:$Rd),
                 (v4i32 (opnode (v4i32 VPR128:$Rn), (v4i32 VPR128:$Rm))))],
              NoItinerary>;
  }
}
multiclass NeonI_3VSame_BHS_sizes<bit u, bits<5> opcode,
                                  string asmop, SDPatternOperator opnode,
                                  bit Commutable = 0>
   : NeonI_3VSame_HS_sizes<u, opcode,  asmop, opnode, Commutable>
{
  let isCommutable = Commutable in {
    def _8B :  NeonI_3VSame<0b0, u, 0b00, opcode,
               (outs VPR64:$Rd), (ins VPR64:$Rn, VPR64:$Rm),
               asmop # "\t$Rd.8b, $Rn.8b, $Rm.8b",
               [(set (v8i8 VPR64:$Rd),
                  (v8i8 (opnode (v8i8 VPR64:$Rn), (v8i8 VPR64:$Rm))))],
               NoItinerary>;

    def _16B : NeonI_3VSame<0b1, u, 0b00, opcode,
               (outs VPR128:$Rd), (ins VPR128:$Rn, VPR128:$Rm),
               asmop # "\t$Rd.16b, $Rn.16b, $Rm.16b",
               [(set (v16i8 VPR128:$Rd),
                  (v16i8 (opnode (v16i8 VPR128:$Rn), (v16i8 VPR128:$Rm))))],
               NoItinerary>;
  }
}

multiclass NeonI_3VSame_BHSD_sizes<bit u, bits<5> opcode,
                                   string asmop, SDPatternOperator opnode,
                                   bit Commutable = 0>
   : NeonI_3VSame_BHS_sizes<u, opcode,  asmop, opnode, Commutable>
{
  let isCommutable = Commutable in {
    def _2D : NeonI_3VSame<0b1, u, 0b11, opcode,
              (outs VPR128:$Rd), (ins VPR128:$Rn, VPR128:$Rm),
              asmop # "\t$Rd.2d, $Rn.2d, $Rm.2d",
              [(set (v2i64 VPR128:$Rd),
                 (v2i64 (opnode (v2i64 VPR128:$Rn), (v2i64 VPR128:$Rm))))],
              NoItinerary>;
  }
}

// Multiclass NeonI_3VSame_SD_sizes: Operand types are floating point types,
// but Result types can be integer or floating point types.
multiclass NeonI_3VSame_SD_sizes<bit u, bit size, bits<5> opcode,
                                 string asmop, SDPatternOperator opnode2S,
                                 SDPatternOperator opnode4S,
                                 SDPatternOperator opnode2D,
                                 ValueType ResTy2S, ValueType ResTy4S,
                                 ValueType ResTy2D, bit Commutable = 0>
{
  let isCommutable = Commutable in {
    def _2S : NeonI_3VSame<0b0, u, {size, 0b0}, opcode,
              (outs VPR64:$Rd), (ins VPR64:$Rn, VPR64:$Rm),
              asmop # "\t$Rd.2s, $Rn.2s, $Rm.2s",
              [(set (ResTy2S VPR64:$Rd),
                 (ResTy2S (opnode2S (v2f32 VPR64:$Rn), (v2f32 VPR64:$Rm))))],
              NoItinerary>;

    def _4S : NeonI_3VSame<0b1, u, {size, 0b0}, opcode,
              (outs VPR128:$Rd), (ins VPR128:$Rn, VPR128:$Rm),
              asmop # "\t$Rd.4s, $Rn.4s, $Rm.4s",
              [(set (ResTy4S VPR128:$Rd),
                 (ResTy4S (opnode4S (v4f32 VPR128:$Rn), (v4f32 VPR128:$Rm))))],
              NoItinerary>;

    def _2D : NeonI_3VSame<0b1, u, {size, 0b1}, opcode,
              (outs VPR128:$Rd), (ins VPR128:$Rn, VPR128:$Rm),
              asmop # "\t$Rd.2d, $Rn.2d, $Rm.2d",
              [(set (ResTy2D VPR128:$Rd),
                 (ResTy2D (opnode2D (v2f64 VPR128:$Rn), (v2f64 VPR128:$Rm))))],
               NoItinerary>;
  }
}

//===----------------------------------------------------------------------===//
// Instruction Definitions
//===----------------------------------------------------------------------===//

// Vector Arithmetic Instructions

// Vector Add (Integer and Floating-Point)

defm ADDvvv :  NeonI_3VSame_BHSD_sizes<0b0, 0b10000, "add", add, 1>;
defm FADDvvv : NeonI_3VSame_SD_sizes<0b0, 0b0, 0b11010, "fadd", fadd, fadd, fadd,
                                     v2f32, v4f32, v2f64, 1>;

// Vector Sub (Integer and Floating-Point)

defm SUBvvv :  NeonI_3VSame_BHSD_sizes<0b1, 0b10000, "sub", sub, 0>;
defm FSUBvvv : NeonI_3VSame_SD_sizes<0b0, 0b1, 0b11010, "fsub", fsub, fsub, fsub,
                                     v2f32, v4f32, v2f64, 0>;

// Vector Multiply (Integer and Floating-Point)

defm MULvvv :  NeonI_3VSame_BHS_sizes<0b0, 0b10011, "mul", mul, 1>;
defm FMULvvv : NeonI_3VSame_SD_sizes<0b1, 0b0, 0b11011, "fmul", fmul, fmul, fmul,
                                     v2f32, v4f32, v2f64, 1>;

// Vector Multiply (Polynomial)

defm PMULvvv : NeonI_3VSame_B_sizes<0b1, 0b00, 0b10011, "pmul",
                                    int_arm_neon_vmulp, int_arm_neon_vmulp, 1>;

// Vector Multiply-accumulate and Multiply-subtract (Integer)

// class NeonI_3VSame_Constraint_impl: NeonI_3VSame with no data type and
// two operands constraints.
class NeonI_3VSame_Constraint_impl<string asmop, string asmlane,
  RegisterClass VPRC, ValueType OpTy, bit q, bit u, bits<2> size, bits<5> opcode,
  SDPatternOperator opnode>
  : NeonI_3VSame<q, u, size, opcode,
    (outs VPRC:$Rd), (ins VPRC:$src, VPRC:$Rn, VPRC:$Rm),
    asmop # "\t$Rd" # asmlane # ", $Rn" # asmlane # ", $Rm" # asmlane,
    [(set (OpTy VPRC:$Rd),
       (OpTy (opnode (OpTy VPRC:$src), (OpTy VPRC:$Rn), (OpTy VPRC:$Rm))))],
    NoItinerary> {
  let Constraints = "$src = $Rd";
}

def Neon_mla : PatFrag<(ops node:$Ra, node:$Rn, node:$Rm),
                       (add node:$Ra, (mul node:$Rn, node:$Rm))>;

def Neon_mls : PatFrag<(ops node:$Ra, node:$Rn, node:$Rm),
                       (sub node:$Ra, (mul node:$Rn, node:$Rm))>;


def MLAvvv_8B:  NeonI_3VSame_Constraint_impl<"mla", ".8b",  VPR64,  v8i8,
                                             0b0, 0b0, 0b00, 0b10010, Neon_mla>;
def MLAvvv_16B: NeonI_3VSame_Constraint_impl<"mla", ".16b", VPR128, v16i8,
                                             0b1, 0b0, 0b00, 0b10010, Neon_mla>;
def MLAvvv_4H:  NeonI_3VSame_Constraint_impl<"mla", ".4h",  VPR64,  v4i16,
                                             0b0, 0b0, 0b01, 0b10010, Neon_mla>;
def MLAvvv_8H:  NeonI_3VSame_Constraint_impl<"mla", ".8h",  VPR128, v8i16,
                                             0b1, 0b0, 0b01, 0b10010, Neon_mla>;
def MLAvvv_2S:  NeonI_3VSame_Constraint_impl<"mla", ".2s",  VPR64,  v2i32,
                                             0b0, 0b0, 0b10, 0b10010, Neon_mla>;
def MLAvvv_4S:  NeonI_3VSame_Constraint_impl<"mla", ".4s",  VPR128, v4i32,
                                             0b1, 0b0, 0b10, 0b10010, Neon_mla>;

def MLSvvv_8B:  NeonI_3VSame_Constraint_impl<"mls", ".8b",  VPR64,  v8i8,
                                             0b0, 0b1, 0b00, 0b10010, Neon_mls>;
def MLSvvv_16B: NeonI_3VSame_Constraint_impl<"mls", ".16b", VPR128, v16i8,
                                             0b1, 0b1, 0b00, 0b10010, Neon_mls>;
def MLSvvv_4H:  NeonI_3VSame_Constraint_impl<"mls", ".4h",  VPR64,  v4i16,
                                             0b0, 0b1, 0b01, 0b10010, Neon_mls>;
def MLSvvv_8H:  NeonI_3VSame_Constraint_impl<"mls", ".8h",  VPR128, v8i16,
                                             0b1, 0b1, 0b01, 0b10010, Neon_mls>;
def MLSvvv_2S:  NeonI_3VSame_Constraint_impl<"mls", ".2s",  VPR64,  v2i32,
                                             0b0, 0b1, 0b10, 0b10010, Neon_mls>;
def MLSvvv_4S:  NeonI_3VSame_Constraint_impl<"mls", ".4s",  VPR128, v4i32,
                                             0b1, 0b1, 0b10, 0b10010, Neon_mls>;

// Vector Multiply-accumulate and Multiply-subtract (Floating Point)

def Neon_fmla : PatFrag<(ops node:$Ra, node:$Rn, node:$Rm),
                        (fadd node:$Ra, (fmul node:$Rn, node:$Rm))>;

def Neon_fmls : PatFrag<(ops node:$Ra, node:$Rn, node:$Rm),
                        (fsub node:$Ra, (fmul node:$Rn, node:$Rm))>;

let Predicates = [HasNEON, UseFusedMAC] in {
def FMLAvvv_2S: NeonI_3VSame_Constraint_impl<"fmla", ".2s",  VPR64,  v2f32,
                                             0b0, 0b0, 0b00, 0b11001, Neon_fmla>;
def FMLAvvv_4S: NeonI_3VSame_Constraint_impl<"fmla", ".4s",  VPR128, v4f32,
                                             0b1, 0b0, 0b00, 0b11001, Neon_fmla>;
def FMLAvvv_2D: NeonI_3VSame_Constraint_impl<"fmla", ".2d",  VPR128, v2f64,
                                             0b1, 0b0, 0b01, 0b11001, Neon_fmla>;

def FMLSvvv_2S: NeonI_3VSame_Constraint_impl<"fmls", ".2s",  VPR64,  v2f32,
                                              0b0, 0b0, 0b10, 0b11001, Neon_fmls>;
def FMLSvvv_4S: NeonI_3VSame_Constraint_impl<"fmls", ".4s",  VPR128, v4f32,
                                             0b1, 0b0, 0b10, 0b11001, Neon_fmls>;
def FMLSvvv_2D: NeonI_3VSame_Constraint_impl<"fmls", ".2d",  VPR128, v2f64,
                                             0b1, 0b0, 0b11, 0b11001, Neon_fmls>;
}

// We're also allowed to match the fma instruction regardless of compile
// options.
def : Pat<(v2f32 (fma VPR64:$Rn, VPR64:$Rm, VPR64:$Ra)),
          (FMLAvvv_2S VPR64:$Ra, VPR64:$Rn, VPR64:$Rm)>;
def : Pat<(v4f32 (fma VPR128:$Rn, VPR128:$Rm, VPR128:$Ra)),
          (FMLAvvv_4S VPR128:$Ra, VPR128:$Rn, VPR128:$Rm)>;
def : Pat<(v2f64 (fma VPR128:$Rn, VPR128:$Rm, VPR128:$Ra)),
          (FMLAvvv_2D VPR128:$Ra, VPR128:$Rn, VPR128:$Rm)>;

def : Pat<(v2f32 (fma (fneg VPR64:$Rn), VPR64:$Rm, VPR64:$Ra)),
          (FMLSvvv_2S VPR64:$Ra, VPR64:$Rn, VPR64:$Rm)>;
def : Pat<(v4f32 (fma (fneg VPR128:$Rn), VPR128:$Rm, VPR128:$Ra)),
          (FMLSvvv_4S VPR128:$Ra, VPR128:$Rn, VPR128:$Rm)>;
def : Pat<(v2f64 (fma (fneg VPR128:$Rn), VPR128:$Rm, VPR128:$Ra)),
          (FMLSvvv_2D VPR128:$Ra, VPR128:$Rn, VPR128:$Rm)>;

// Vector Divide (Floating-Point)

defm FDIVvvv : NeonI_3VSame_SD_sizes<0b1, 0b0, 0b11111, "fdiv", fdiv, fdiv, fdiv,
                                     v2f32, v4f32, v2f64, 0>;

// Vector Bitwise Operations

// Vector Bitwise AND

defm ANDvvv : NeonI_3VSame_B_sizes<0b0, 0b00, 0b00011, "and", and, and, 1>;

// Vector Bitwise Exclusive OR

defm EORvvv : NeonI_3VSame_B_sizes<0b1, 0b00, 0b00011, "eor", xor, xor, 1>;

// Vector Bitwise OR

defm ORRvvv : NeonI_3VSame_B_sizes<0b0, 0b10, 0b00011, "orr", or, or, 1>;

// ORR disassembled as MOV if Vn==Vm

// Vector Move - register
// Alias for ORR if Vn=Vm and it is the preferred syntax
def : NeonInstAlias<"mov $Rd.8b, $Rn.8b",
                    (ORRvvv_8B VPR64:$Rd, VPR64:$Rn, VPR64:$Rn)>;
def : NeonInstAlias<"mov $Rd.16b, $Rn.16b",
                    (ORRvvv_16B VPR128:$Rd, VPR128:$Rn, VPR128:$Rn)>;

def Neon_immAllOnes: PatLeaf<(Neon_movi (i32 timm), (i32 imm)), [{
  ConstantSDNode *ImmConstVal = cast<ConstantSDNode>(N->getOperand(0));
  ConstantSDNode *OpCmodeConstVal = cast<ConstantSDNode>(N->getOperand(1));
  unsigned EltBits;
  uint64_t EltVal = A64Imms::decodeNeonModImm(ImmConstVal->getZExtValue(),
    OpCmodeConstVal->getZExtValue(), EltBits);
  return (EltBits == 8 && EltVal == 0xff);
}]>;


def Neon_not8B  : PatFrag<(ops node:$in),
                          (xor node:$in, (bitconvert (v8i8 Neon_immAllOnes)))>;
def Neon_not16B : PatFrag<(ops node:$in),
                          (xor node:$in, (bitconvert (v16i8 Neon_immAllOnes)))>;

def Neon_orn8B : PatFrag<(ops node:$Rn, node:$Rm),
                         (or node:$Rn, (Neon_not8B node:$Rm))>;

def Neon_orn16B : PatFrag<(ops node:$Rn, node:$Rm),
                          (or node:$Rn, (Neon_not16B node:$Rm))>;

def Neon_bic8B : PatFrag<(ops node:$Rn, node:$Rm),
                         (and node:$Rn, (Neon_not8B node:$Rm))>;

def Neon_bic16B : PatFrag<(ops node:$Rn, node:$Rm),
                          (and node:$Rn, (Neon_not16B node:$Rm))>;


// Vector Bitwise OR NOT - register

defm ORNvvv : NeonI_3VSame_B_sizes<0b0, 0b11, 0b00011, "orn",
                                   Neon_orn8B, Neon_orn16B, 0>;

// Vector Bitwise Bit Clear (AND NOT) - register

defm BICvvv : NeonI_3VSame_B_sizes<0b0, 0b01, 0b00011, "bic",
                                   Neon_bic8B, Neon_bic16B, 0>;

multiclass Neon_bitwise2V_patterns<SDPatternOperator opnode8B,
                                   SDPatternOperator opnode16B,
                                   Instruction INST8B,
                                   Instruction INST16B> {
  def : Pat<(v2i32 (opnode8B VPR64:$Rn, VPR64:$Rm)),
            (INST8B VPR64:$Rn, VPR64:$Rm)>;
  def : Pat<(v4i16 (opnode8B VPR64:$Rn, VPR64:$Rm)),
            (INST8B VPR64:$Rn, VPR64:$Rm)>;
  def : Pat<(v1i64 (opnode8B VPR64:$Rn, VPR64:$Rm)),
            (INST8B VPR64:$Rn, VPR64:$Rm)>;
  def : Pat<(v4i32 (opnode16B VPR128:$Rn, VPR128:$Rm)),
            (INST16B VPR128:$Rn, VPR128:$Rm)>;
  def : Pat<(v8i16 (opnode16B VPR128:$Rn, VPR128:$Rm)),
            (INST16B VPR128:$Rn, VPR128:$Rm)>;
  def : Pat<(v2i64 (opnode16B VPR128:$Rn, VPR128:$Rm)),
            (INST16B VPR128:$Rn, VPR128:$Rm)>;
}

// Additional patterns for bitwise instructions AND, EOR, ORR, BIC, ORN
defm : Neon_bitwise2V_patterns<and, and, ANDvvv_8B, ANDvvv_16B>;
defm : Neon_bitwise2V_patterns<or,  or,  ORRvvv_8B, ORRvvv_16B>;
defm : Neon_bitwise2V_patterns<xor, xor, EORvvv_8B, EORvvv_16B>;
defm : Neon_bitwise2V_patterns<Neon_bic8B, Neon_bic16B, BICvvv_8B, BICvvv_16B>;
defm : Neon_bitwise2V_patterns<Neon_orn8B, Neon_orn16B, ORNvvv_8B, ORNvvv_16B>;

//   Vector Bitwise Select
def BSLvvv_8B  : NeonI_3VSame_Constraint_impl<"bsl", ".8b",  VPR64, v8i8,
                                              0b0, 0b1, 0b01, 0b00011, Neon_bsl>;

def BSLvvv_16B : NeonI_3VSame_Constraint_impl<"bsl", ".16b", VPR128, v16i8,
                                              0b1, 0b1, 0b01, 0b00011, Neon_bsl>;

multiclass Neon_bitwise3V_patterns<SDPatternOperator opnode,
                                   Instruction INST8B,
                                   Instruction INST16B> {
  // Disassociate type from instruction definition
  def : Pat<(v2i32 (opnode VPR64:$src,VPR64:$Rn, VPR64:$Rm)),
            (INST8B VPR64:$src, VPR64:$Rn, VPR64:$Rm)>;
  def : Pat<(v4i16 (opnode VPR64:$src, VPR64:$Rn, VPR64:$Rm)),
            (INST8B VPR64:$src, VPR64:$Rn, VPR64:$Rm)>;
  def : Pat<(v1i64 (opnode VPR64:$src, VPR64:$Rn, VPR64:$Rm)),
            (INST8B VPR64:$src, VPR64:$Rn, VPR64:$Rm)>;
  def : Pat<(v4i32 (opnode VPR128:$src, VPR128:$Rn, VPR128:$Rm)),
            (INST16B VPR128:$src, VPR128:$Rn, VPR128:$Rm)>;
  def : Pat<(v8i16 (opnode VPR128:$src, VPR128:$Rn, VPR128:$Rm)),
            (INST16B VPR128:$src, VPR128:$Rn, VPR128:$Rm)>;
  def : Pat<(v2i64 (opnode VPR128:$src, VPR128:$Rn, VPR128:$Rm)),
            (INST16B VPR128:$src, VPR128:$Rn, VPR128:$Rm)>;

  // Allow to match BSL instruction pattern with non-constant operand
  def : Pat<(v8i8 (or (and VPR64:$Rn, VPR64:$Rd),
                    (and VPR64:$Rm, (Neon_not8B VPR64:$Rd)))),
          (INST8B VPR64:$Rd, VPR64:$Rn, VPR64:$Rm)>;
  def : Pat<(v4i16 (or (and VPR64:$Rn, VPR64:$Rd),
                     (and VPR64:$Rm, (Neon_not8B VPR64:$Rd)))),
          (INST8B VPR64:$Rd, VPR64:$Rn, VPR64:$Rm)>;
  def : Pat<(v2i32 (or (and VPR64:$Rn, VPR64:$Rd),
                     (and VPR64:$Rm, (Neon_not8B VPR64:$Rd)))),
          (INST8B VPR64:$Rd, VPR64:$Rn, VPR64:$Rm)>;
  def : Pat<(v1i64 (or (and VPR64:$Rn, VPR64:$Rd),
                     (and VPR64:$Rm, (Neon_not8B VPR64:$Rd)))),
          (INST8B VPR64:$Rd, VPR64:$Rn, VPR64:$Rm)>;
  def : Pat<(v16i8 (or (and VPR128:$Rn, VPR128:$Rd),
                     (and VPR128:$Rm, (Neon_not16B VPR128:$Rd)))),
          (INST16B VPR128:$Rd, VPR128:$Rn, VPR128:$Rm)>;
  def : Pat<(v8i16 (or (and VPR128:$Rn, VPR128:$Rd),
                     (and VPR128:$Rm, (Neon_not16B VPR128:$Rd)))),
          (INST16B VPR128:$Rd, VPR128:$Rn, VPR128:$Rm)>;
  def : Pat<(v4i32 (or (and VPR128:$Rn, VPR128:$Rd),
                     (and VPR128:$Rm, (Neon_not16B VPR128:$Rd)))),
          (INST16B VPR128:$Rd, VPR128:$Rn, VPR128:$Rm)>;
  def : Pat<(v2i64 (or (and VPR128:$Rn, VPR128:$Rd),
                     (and VPR128:$Rm, (Neon_not16B VPR128:$Rd)))),
          (INST16B VPR128:$Rd, VPR128:$Rn, VPR128:$Rm)>;

  // Allow to match llvm.arm.* intrinsics.
  def : Pat<(v8i8 (int_arm_neon_vbsl (v8i8 VPR64:$src),
                    (v8i8 VPR64:$Rn), (v8i8 VPR64:$Rm))),
            (INST8B VPR64:$src, VPR64:$Rn, VPR64:$Rm)>;
  def : Pat<(v4i16 (int_arm_neon_vbsl (v4i16 VPR64:$src),
                    (v4i16 VPR64:$Rn), (v4i16 VPR64:$Rm))),
            (INST8B VPR64:$src, VPR64:$Rn, VPR64:$Rm)>;
  def : Pat<(v2i32 (int_arm_neon_vbsl (v2i32 VPR64:$src),
                    (v2i32 VPR64:$Rn), (v2i32 VPR64:$Rm))),
            (INST8B VPR64:$src, VPR64:$Rn, VPR64:$Rm)>;
  def : Pat<(v1i64 (int_arm_neon_vbsl (v1i64 VPR64:$src),
                    (v1i64 VPR64:$Rn), (v1i64 VPR64:$Rm))),
            (INST8B VPR64:$src, VPR64:$Rn, VPR64:$Rm)>;
  def : Pat<(v2f32 (int_arm_neon_vbsl (v2f32 VPR64:$src),
                    (v2f32 VPR64:$Rn), (v2f32 VPR64:$Rm))),
            (INST8B VPR64:$src, VPR64:$Rn, VPR64:$Rm)>;
  def : Pat<(v16i8 (int_arm_neon_vbsl (v16i8 VPR128:$src),
                    (v16i8 VPR128:$Rn), (v16i8 VPR128:$Rm))),
            (INST16B VPR128:$src, VPR128:$Rn, VPR128:$Rm)>;
  def : Pat<(v8i16 (int_arm_neon_vbsl (v8i16 VPR128:$src),
                    (v8i16 VPR128:$Rn), (v8i16 VPR128:$Rm))),
            (INST16B VPR128:$src, VPR128:$Rn, VPR128:$Rm)>;
  def : Pat<(v4i32 (int_arm_neon_vbsl (v4i32 VPR128:$src),
                    (v4i32 VPR128:$Rn), (v4i32 VPR128:$Rm))),
            (INST16B VPR128:$src, VPR128:$Rn, VPR128:$Rm)>;
  def : Pat<(v2i64 (int_arm_neon_vbsl (v2i64 VPR128:$src),
                    (v2i64 VPR128:$Rn), (v2i64 VPR128:$Rm))),
            (INST16B VPR128:$src, VPR128:$Rn, VPR128:$Rm)>;
  def : Pat<(v4f32 (int_arm_neon_vbsl (v4f32 VPR128:$src),
                    (v4f32 VPR128:$Rn), (v4f32 VPR128:$Rm))),
            (INST16B VPR128:$src, VPR128:$Rn, VPR128:$Rm)>;
  def : Pat<(v2f64 (int_arm_neon_vbsl (v2f64 VPR128:$src),
                    (v2f64 VPR128:$Rn), (v2f64 VPR128:$Rm))),
            (INST16B VPR128:$src, VPR128:$Rn, VPR128:$Rm)>;
}

// Additional patterns for bitwise instruction BSL
defm: Neon_bitwise3V_patterns<Neon_bsl, BSLvvv_8B, BSLvvv_16B>;

def Neon_NoBSLop : PatFrag<(ops node:$src, node:$Rn, node:$Rm),
                           (Neon_bsl node:$src, node:$Rn, node:$Rm),
                           [{ (void)N; return false; }]>;

// Vector Bitwise Insert if True

def BITvvv_8B  : NeonI_3VSame_Constraint_impl<"bit", ".8b", VPR64,   v8i8,
                   0b0, 0b1, 0b10, 0b00011, Neon_NoBSLop>;
def BITvvv_16B : NeonI_3VSame_Constraint_impl<"bit", ".16b", VPR128, v16i8,
                   0b1, 0b1, 0b10, 0b00011, Neon_NoBSLop>;

// Vector Bitwise Insert if False

def BIFvvv_8B  : NeonI_3VSame_Constraint_impl<"bif", ".8b", VPR64,  v8i8,
                                0b0, 0b1, 0b11, 0b00011, Neon_NoBSLop>;
def BIFvvv_16B : NeonI_3VSame_Constraint_impl<"bif", ".16b", VPR128, v16i8,
                                0b1, 0b1, 0b11, 0b00011, Neon_NoBSLop>;

// Vector Absolute Difference and Accumulate (Signed, Unsigned)

def Neon_uaba : PatFrag<(ops node:$Ra, node:$Rn, node:$Rm),
                       (add node:$Ra, (int_arm_neon_vabdu node:$Rn, node:$Rm))>;
def Neon_saba : PatFrag<(ops node:$Ra, node:$Rn, node:$Rm),
                       (add node:$Ra, (int_arm_neon_vabds node:$Rn, node:$Rm))>;

// Vector Absolute Difference and Accumulate (Unsigned)
def UABAvvv_8B :  NeonI_3VSame_Constraint_impl<"uaba", ".8b",  VPR64,  v8i8,
                    0b0, 0b1, 0b00, 0b01111, Neon_uaba>;
def UABAvvv_16B : NeonI_3VSame_Constraint_impl<"uaba", ".16b", VPR128, v16i8,
                    0b1, 0b1, 0b00, 0b01111, Neon_uaba>;
def UABAvvv_4H :  NeonI_3VSame_Constraint_impl<"uaba", ".4h",  VPR64,  v4i16,
                    0b0, 0b1, 0b01, 0b01111, Neon_uaba>;
def UABAvvv_8H :  NeonI_3VSame_Constraint_impl<"uaba", ".8h",  VPR128, v8i16,
                    0b1, 0b1, 0b01, 0b01111, Neon_uaba>;
def UABAvvv_2S :  NeonI_3VSame_Constraint_impl<"uaba", ".2s",  VPR64,  v2i32,
                    0b0, 0b1, 0b10, 0b01111, Neon_uaba>;
def UABAvvv_4S :  NeonI_3VSame_Constraint_impl<"uaba", ".4s",  VPR128, v4i32,
                    0b1, 0b1, 0b10, 0b01111, Neon_uaba>;

// Vector Absolute Difference and Accumulate (Signed)
def SABAvvv_8B :  NeonI_3VSame_Constraint_impl<"saba", ".8b",  VPR64,  v8i8,
                    0b0, 0b0, 0b00, 0b01111, Neon_saba>;
def SABAvvv_16B : NeonI_3VSame_Constraint_impl<"saba", ".16b", VPR128, v16i8,
                    0b1, 0b0, 0b00, 0b01111, Neon_saba>;
def SABAvvv_4H :  NeonI_3VSame_Constraint_impl<"saba", ".4h",  VPR64,  v4i16,
                    0b0, 0b0, 0b01, 0b01111, Neon_saba>;
def SABAvvv_8H :  NeonI_3VSame_Constraint_impl<"saba", ".8h",  VPR128, v8i16,
                    0b1, 0b0, 0b01, 0b01111, Neon_saba>;
def SABAvvv_2S :  NeonI_3VSame_Constraint_impl<"saba", ".2s",  VPR64,  v2i32,
                    0b0, 0b0, 0b10, 0b01111, Neon_saba>;
def SABAvvv_4S :  NeonI_3VSame_Constraint_impl<"saba", ".4s",  VPR128, v4i32,
                    0b1, 0b0, 0b10, 0b01111, Neon_saba>;


// Vector Absolute Difference (Signed, Unsigned)
defm UABDvvv : NeonI_3VSame_BHS_sizes<0b1, 0b01110, "uabd", int_arm_neon_vabdu, 0>;
defm SABDvvv : NeonI_3VSame_BHS_sizes<0b0, 0b01110, "sabd", int_arm_neon_vabds, 0>;

// Vector Absolute Difference (Floating Point)
defm FABDvvv: NeonI_3VSame_SD_sizes<0b1, 0b1, 0b11010, "fabd",
                                    int_arm_neon_vabds, int_arm_neon_vabds,
                                    int_arm_neon_vabds, v2f32, v4f32, v2f64, 0>;

// Vector Reciprocal Step (Floating Point)
defm FRECPSvvv : NeonI_3VSame_SD_sizes<0b0, 0b0, 0b11111, "frecps",
                                       int_arm_neon_vrecps, int_arm_neon_vrecps,
                                       int_arm_neon_vrecps,
                                       v2f32, v4f32, v2f64, 0>;

// Vector Reciprocal Square Root Step (Floating Point)
defm FRSQRTSvvv : NeonI_3VSame_SD_sizes<0b0, 0b1, 0b11111, "frsqrts",
                                        int_arm_neon_vrsqrts,
                                        int_arm_neon_vrsqrts,
                                        int_arm_neon_vrsqrts,
                                        v2f32, v4f32, v2f64, 0>;

// Vector Comparisons

def Neon_cmeq : PatFrag<(ops node:$lhs, node:$rhs),
                        (Neon_cmp node:$lhs, node:$rhs, SETEQ)>;
def Neon_cmphs : PatFrag<(ops node:$lhs, node:$rhs),
                         (Neon_cmp node:$lhs, node:$rhs, SETUGE)>;
def Neon_cmge : PatFrag<(ops node:$lhs, node:$rhs),
                        (Neon_cmp node:$lhs, node:$rhs, SETGE)>;
def Neon_cmhi : PatFrag<(ops node:$lhs, node:$rhs),
                        (Neon_cmp node:$lhs, node:$rhs, SETUGT)>;
def Neon_cmgt : PatFrag<(ops node:$lhs, node:$rhs),
                        (Neon_cmp node:$lhs, node:$rhs, SETGT)>;

// NeonI_compare_aliases class: swaps register operands to implement
// comparison aliases, e.g., CMLE is alias for CMGE with operands reversed.
class NeonI_compare_aliases<string asmop, string asmlane,
                            Instruction inst, RegisterClass VPRC>
  : NeonInstAlias<asmop # "\t$Rd" # asmlane #", $Rn" # asmlane #
                    ", $Rm" # asmlane,
                  (inst VPRC:$Rd, VPRC:$Rm, VPRC:$Rn), 0b0>;

// Vector Comparisons (Integer)

// Vector Compare Mask Equal (Integer)
let isCommutable =1 in {
defm CMEQvvv : NeonI_3VSame_BHSD_sizes<0b1, 0b10001, "cmeq", Neon_cmeq, 0>;
}

// Vector Compare Mask Higher or Same (Unsigned Integer)
defm CMHSvvv : NeonI_3VSame_BHSD_sizes<0b1, 0b00111, "cmhs", Neon_cmphs, 0>;

// Vector Compare Mask Greater Than or Equal (Integer)
defm CMGEvvv : NeonI_3VSame_BHSD_sizes<0b0, 0b00111, "cmge", Neon_cmge, 0>;

// Vector Compare Mask Higher (Unsigned Integer)
defm CMHIvvv : NeonI_3VSame_BHSD_sizes<0b1, 0b00110, "cmhi", Neon_cmhi, 0>;

// Vector Compare Mask Greater Than (Integer)
defm CMGTvvv : NeonI_3VSame_BHSD_sizes<0b0, 0b00110, "cmgt", Neon_cmgt, 0>;

// Vector Compare Mask Bitwise Test (Integer)
defm CMTSTvvv:  NeonI_3VSame_BHSD_sizes<0b0, 0b10001, "cmtst", Neon_tst, 0>;

// Vector Compare Mask Less or Same (Unsigned Integer)
// CMLS is alias for CMHS with operands reversed.
def CMLSvvv_8B  : NeonI_compare_aliases<"cmls", ".8b",  CMHSvvv_8B,  VPR64>;
def CMLSvvv_16B : NeonI_compare_aliases<"cmls", ".16b", CMHSvvv_16B, VPR128>;
def CMLSvvv_4H  : NeonI_compare_aliases<"cmls", ".4h",  CMHSvvv_4H,  VPR64>;
def CMLSvvv_8H  : NeonI_compare_aliases<"cmls", ".8h",  CMHSvvv_8H,  VPR128>;
def CMLSvvv_2S  : NeonI_compare_aliases<"cmls", ".2s",  CMHSvvv_2S,  VPR64>;
def CMLSvvv_4S  : NeonI_compare_aliases<"cmls", ".4s",  CMHSvvv_4S,  VPR128>;
def CMLSvvv_2D  : NeonI_compare_aliases<"cmls", ".2d",  CMHSvvv_2D,  VPR128>;

// Vector Compare Mask Less Than or Equal (Integer)
// CMLE is alias for CMGE with operands reversed.
def CMLEvvv_8B  : NeonI_compare_aliases<"cmle", ".8b",  CMGEvvv_8B,  VPR64>;
def CMLEvvv_16B : NeonI_compare_aliases<"cmle", ".16b", CMGEvvv_16B, VPR128>;
def CMLEvvv_4H  : NeonI_compare_aliases<"cmle", ".4h",  CMGEvvv_4H,  VPR64>;
def CMLEvvv_8H  : NeonI_compare_aliases<"cmle", ".8h",  CMGEvvv_8H,  VPR128>;
def CMLEvvv_2S  : NeonI_compare_aliases<"cmle", ".2s",  CMGEvvv_2S,  VPR64>;
def CMLEvvv_4S  : NeonI_compare_aliases<"cmle", ".4s",  CMGEvvv_4S,  VPR128>;
def CMLEvvv_2D  : NeonI_compare_aliases<"cmle", ".2d",  CMGEvvv_2D,  VPR128>;

// Vector Compare Mask Lower (Unsigned Integer)
// CMLO is alias for CMHI with operands reversed.
def CMLOvvv_8B  : NeonI_compare_aliases<"cmlo", ".8b",  CMHIvvv_8B,  VPR64>;
def CMLOvvv_16B : NeonI_compare_aliases<"cmlo", ".16b", CMHIvvv_16B, VPR128>;
def CMLOvvv_4H  : NeonI_compare_aliases<"cmlo", ".4h",  CMHIvvv_4H,  VPR64>;
def CMLOvvv_8H  : NeonI_compare_aliases<"cmlo", ".8h",  CMHIvvv_8H,  VPR128>;
def CMLOvvv_2S  : NeonI_compare_aliases<"cmlo", ".2s",  CMHIvvv_2S,  VPR64>;
def CMLOvvv_4S  : NeonI_compare_aliases<"cmlo", ".4s",  CMHIvvv_4S,  VPR128>;
def CMLOvvv_2D  : NeonI_compare_aliases<"cmlo", ".2d",  CMHIvvv_2D,  VPR128>;

// Vector Compare Mask Less Than (Integer)
// CMLT is alias for CMGT with operands reversed.
def CMLTvvv_8B  : NeonI_compare_aliases<"cmlt", ".8b",  CMGTvvv_8B,  VPR64>;
def CMLTvvv_16B : NeonI_compare_aliases<"cmlt", ".16b", CMGTvvv_16B, VPR128>;
def CMLTvvv_4H  : NeonI_compare_aliases<"cmlt", ".4h",  CMGTvvv_4H,  VPR64>;
def CMLTvvv_8H  : NeonI_compare_aliases<"cmlt", ".8h",  CMGTvvv_8H,  VPR128>;
def CMLTvvv_2S  : NeonI_compare_aliases<"cmlt", ".2s",  CMGTvvv_2S,  VPR64>;
def CMLTvvv_4S  : NeonI_compare_aliases<"cmlt", ".4s",  CMGTvvv_4S,  VPR128>;
def CMLTvvv_2D  : NeonI_compare_aliases<"cmlt", ".2d",  CMGTvvv_2D,  VPR128>;


def neon_uimm0_asmoperand : AsmOperandClass
{
  let Name = "UImm0";
  let PredicateMethod = "isUImm<0>";
  let RenderMethod = "addImmOperands";
}

def neon_uimm0 : Operand<i32>, ImmLeaf<i32, [{return Imm == 0;}]> {
  let ParserMatchClass = neon_uimm0_asmoperand;
  let PrintMethod = "printNeonUImm0Operand";

}

multiclass NeonI_cmpz_sizes<bit u, bits<5> opcode, string asmop, CondCode CC>
{
  def _8B :  NeonI_2VMisc<0b0, u, 0b00, opcode,
             (outs VPR64:$Rd), (ins VPR64:$Rn, neon_uimm0:$Imm),
             asmop # "\t$Rd.8b, $Rn.8b, $Imm",
             [(set (v8i8 VPR64:$Rd),
                (v8i8 (Neon_cmpz (v8i8 VPR64:$Rn), (i32 imm:$Imm), CC)))],
             NoItinerary>;

  def _16B : NeonI_2VMisc<0b1, u, 0b00, opcode,
             (outs VPR128:$Rd), (ins VPR128:$Rn, neon_uimm0:$Imm),
             asmop # "\t$Rd.16b, $Rn.16b, $Imm",
             [(set (v16i8 VPR128:$Rd),
                (v16i8 (Neon_cmpz (v16i8 VPR128:$Rn), (i32 imm:$Imm), CC)))],
             NoItinerary>;

  def _4H : NeonI_2VMisc<0b0, u, 0b01, opcode,
            (outs VPR64:$Rd), (ins VPR64:$Rn, neon_uimm0:$Imm),
            asmop # "\t$Rd.4h, $Rn.4h, $Imm",
            [(set (v4i16 VPR64:$Rd),
               (v4i16 (Neon_cmpz (v4i16 VPR64:$Rn), (i32 imm:$Imm), CC)))],
            NoItinerary>;

  def _8H : NeonI_2VMisc<0b1, u, 0b01, opcode,
            (outs VPR128:$Rd), (ins VPR128:$Rn, neon_uimm0:$Imm),
            asmop # "\t$Rd.8h, $Rn.8h, $Imm",
            [(set (v8i16 VPR128:$Rd),
               (v8i16 (Neon_cmpz (v8i16 VPR128:$Rn), (i32 imm:$Imm), CC)))],
            NoItinerary>;

  def _2S : NeonI_2VMisc<0b0, u, 0b10, opcode,
            (outs VPR64:$Rd), (ins VPR64:$Rn, neon_uimm0:$Imm),
            asmop # "\t$Rd.2s, $Rn.2s, $Imm",
            [(set (v2i32 VPR64:$Rd),
               (v2i32 (Neon_cmpz (v2i32 VPR64:$Rn), (i32 imm:$Imm), CC)))],
            NoItinerary>;

  def _4S : NeonI_2VMisc<0b1, u, 0b10, opcode,
            (outs VPR128:$Rd), (ins VPR128:$Rn, neon_uimm0:$Imm),
            asmop # "\t$Rd.4s, $Rn.4s, $Imm",
            [(set (v4i32 VPR128:$Rd),
               (v4i32 (Neon_cmpz (v4i32 VPR128:$Rn), (i32 imm:$Imm), CC)))],
            NoItinerary>;

  def _2D : NeonI_2VMisc<0b1, u, 0b11, opcode,
            (outs VPR128:$Rd), (ins VPR128:$Rn, neon_uimm0:$Imm),
            asmop # "\t$Rd.2d, $Rn.2d, $Imm",
            [(set (v2i64 VPR128:$Rd),
               (v2i64 (Neon_cmpz (v2i64 VPR128:$Rn), (i32 imm:$Imm), CC)))],
            NoItinerary>;
}

// Vector Compare Mask Equal to Zero (Integer)
defm CMEQvvi : NeonI_cmpz_sizes<0b0, 0b01001, "cmeq", SETEQ>;

// Vector Compare Mask Greater Than or Equal to Zero (Signed Integer)
defm CMGEvvi : NeonI_cmpz_sizes<0b1, 0b01000, "cmge", SETGE>;

// Vector Compare Mask Greater Than Zero (Signed Integer)
defm CMGTvvi : NeonI_cmpz_sizes<0b0, 0b01000, "cmgt", SETGT>;

// Vector Compare Mask Less Than or Equal To Zero (Signed Integer)
defm CMLEvvi : NeonI_cmpz_sizes<0b1, 0b01001, "cmle", SETLE>;

// Vector Compare Mask Less Than Zero (Signed Integer)
defm CMLTvvi : NeonI_cmpz_sizes<0b0, 0b01010, "cmlt", SETLT>;

// Vector Comparisons (Floating Point)

// Vector Compare Mask Equal (Floating Point)
let isCommutable =1 in {
defm FCMEQvvv : NeonI_3VSame_SD_sizes<0b0, 0b0, 0b11100, "fcmeq", Neon_cmeq,
                                      Neon_cmeq, Neon_cmeq,
                                      v2i32, v4i32, v2i64, 0>;
}

// Vector Compare Mask Greater Than Or Equal (Floating Point)
defm FCMGEvvv : NeonI_3VSame_SD_sizes<0b1, 0b0, 0b11100, "fcmge", Neon_cmge,
                                      Neon_cmge, Neon_cmge,
                                      v2i32, v4i32, v2i64, 0>;

// Vector Compare Mask Greater Than (Floating Point)
defm FCMGTvvv : NeonI_3VSame_SD_sizes<0b1, 0b1, 0b11100, "fcmgt", Neon_cmgt,
                                      Neon_cmgt, Neon_cmgt,
                                      v2i32, v4i32, v2i64, 0>;

// Vector Compare Mask Less Than Or Equal (Floating Point)
// FCMLE is alias for FCMGE with operands reversed.
def FCMLEvvv_2S  : NeonI_compare_aliases<"fcmle", ".2s",  FCMGEvvv_2S,  VPR64>;
def FCMLEvvv_4S  : NeonI_compare_aliases<"fcmle", ".4s",  FCMGEvvv_4S,  VPR128>;
def FCMLEvvv_2D  : NeonI_compare_aliases<"fcmle", ".2d",  FCMGEvvv_2D,  VPR128>;

// Vector Compare Mask Less Than (Floating Point)
// FCMLT is alias for FCMGT with operands reversed.
def FCMLTvvv_2S  : NeonI_compare_aliases<"fcmlt", ".2s",  FCMGTvvv_2S,  VPR64>;
def FCMLTvvv_4S  : NeonI_compare_aliases<"fcmlt", ".4s",  FCMGTvvv_4S,  VPR128>;
def FCMLTvvv_2D  : NeonI_compare_aliases<"fcmlt", ".2d",  FCMGTvvv_2D,  VPR128>;


multiclass NeonI_fpcmpz_sizes<bit u, bit size, bits<5> opcode,
                              string asmop, CondCode CC>
{
  def _2S : NeonI_2VMisc<0b0, u, {size, 0b0}, opcode,
            (outs VPR64:$Rd), (ins VPR64:$Rn, fpz32:$FPImm),
            asmop # "\t$Rd.2s, $Rn.2s, $FPImm",
            [(set (v2i32 VPR64:$Rd),
               (v2i32 (Neon_cmpz (v2f32 VPR64:$Rn), (f32 fpimm:$FPImm), CC)))],
            NoItinerary>;

  def _4S : NeonI_2VMisc<0b1, u, {size, 0b0}, opcode,
            (outs VPR128:$Rd), (ins VPR128:$Rn, fpz32:$FPImm),
            asmop # "\t$Rd.4s, $Rn.4s, $FPImm",
            [(set (v4i32 VPR128:$Rd),
               (v4i32 (Neon_cmpz (v4f32 VPR128:$Rn), (f32 fpimm:$FPImm), CC)))],
            NoItinerary>;

  def _2D : NeonI_2VMisc<0b1, u, {size, 0b1}, opcode,
            (outs VPR128:$Rd), (ins VPR128:$Rn, fpz32:$FPImm),
            asmop # "\t$Rd.2d, $Rn.2d, $FPImm",
            [(set (v2i64 VPR128:$Rd),
               (v2i64 (Neon_cmpz (v2f64 VPR128:$Rn), (f32 fpimm:$FPImm), CC)))],
            NoItinerary>;
}

// Vector Compare Mask Equal to Zero (Floating Point)
defm FCMEQvvi : NeonI_fpcmpz_sizes<0b0, 0b1, 0b01101, "fcmeq", SETEQ>;

// Vector Compare Mask Greater Than or Equal to Zero (Floating Point)
defm FCMGEvvi : NeonI_fpcmpz_sizes<0b1, 0b1, 0b01100, "fcmge", SETGE>;

// Vector Compare Mask Greater Than Zero (Floating Point)
defm FCMGTvvi : NeonI_fpcmpz_sizes<0b0, 0b1, 0b01100, "fcmgt", SETGT>;

// Vector Compare Mask Less Than or Equal To Zero (Floating Point)
defm FCMLEvvi : NeonI_fpcmpz_sizes<0b1, 0b1, 0b01101, "fcmle", SETLE>;

// Vector Compare Mask Less Than Zero (Floating Point)
defm FCMLTvvi : NeonI_fpcmpz_sizes<0b0, 0b1, 0b01110, "fcmlt", SETLT>;

// Vector Absolute Comparisons (Floating Point)

// Vector Absolute Compare Mask Greater Than Or Equal (Floating Point)
defm FACGEvvv : NeonI_3VSame_SD_sizes<0b1, 0b0, 0b11101, "facge",
                                      int_arm_neon_vacged, int_arm_neon_vacgeq,
                                      int_aarch64_neon_vacgeq,
                                      v2i32, v4i32, v2i64, 0>;

// Vector Absolute Compare Mask Greater Than (Floating Point)
defm FACGTvvv : NeonI_3VSame_SD_sizes<0b1, 0b1, 0b11101, "facgt",
                                      int_arm_neon_vacgtd, int_arm_neon_vacgtq,
                                      int_aarch64_neon_vacgtq,
                                      v2i32, v4i32, v2i64, 0>;

// Vector Absolute Compare Mask Less Than Or Equal (Floating Point)
// FACLE is alias for FACGE with operands reversed.
def FACLEvvv_2S  : NeonI_compare_aliases<"facle", ".2s",  FACGEvvv_2S,  VPR64>;
def FACLEvvv_4S  : NeonI_compare_aliases<"facle", ".4s",  FACGEvvv_4S,  VPR128>;
def FACLEvvv_2D  : NeonI_compare_aliases<"facle", ".2d",  FACGEvvv_2D,  VPR128>;

// Vector Absolute Compare Mask Less Than (Floating Point)
// FACLT is alias for FACGT with operands reversed.
def FACLTvvv_2S  : NeonI_compare_aliases<"faclt", ".2s",  FACGTvvv_2S,  VPR64>;
def FACLTvvv_4S  : NeonI_compare_aliases<"faclt", ".4s",  FACGTvvv_4S,  VPR128>;
def FACLTvvv_2D  : NeonI_compare_aliases<"faclt", ".2d",  FACGTvvv_2D,  VPR128>;

// Vector halving add (Integer Signed, Unsigned)
defm SHADDvvv :  NeonI_3VSame_BHS_sizes<0b0, 0b00000, "shadd",
                                        int_arm_neon_vhadds, 1>;
defm UHADDvvv :  NeonI_3VSame_BHS_sizes<0b1, 0b00000, "uhadd",
                                        int_arm_neon_vhaddu, 1>;

// Vector halving sub (Integer Signed, Unsigned)
defm SHSUBvvv :  NeonI_3VSame_BHS_sizes<0b0, 0b00100, "shsub",
                                        int_arm_neon_vhsubs, 0>;
defm UHSUBvvv :  NeonI_3VSame_BHS_sizes<0b1, 0b00100, "uhsub",
                                        int_arm_neon_vhsubu, 0>;

// Vector rouding halving add (Integer Signed, Unsigned)
defm SRHADDvvv :  NeonI_3VSame_BHS_sizes<0b0, 0b00010, "srhadd",
                                         int_arm_neon_vrhadds, 1>;
defm URHADDvvv :  NeonI_3VSame_BHS_sizes<0b1, 0b00010, "urhadd",
                                         int_arm_neon_vrhaddu, 1>;

// Vector Saturating add (Integer Signed, Unsigned)
defm SQADDvvv :  NeonI_3VSame_BHSD_sizes<0b0, 0b00001, "sqadd",
                   int_arm_neon_vqadds, 1>;
defm UQADDvvv :  NeonI_3VSame_BHSD_sizes<0b1, 0b00001, "uqadd",
                   int_arm_neon_vqaddu, 1>;

// Vector Saturating sub (Integer Signed, Unsigned)
defm SQSUBvvv :  NeonI_3VSame_BHSD_sizes<0b0, 0b00101, "sqsub",
                   int_arm_neon_vqsubs, 1>;
defm UQSUBvvv :  NeonI_3VSame_BHSD_sizes<0b1, 0b00101, "uqsub",
                   int_arm_neon_vqsubu, 1>;

// Vector Shift Left (Signed and Unsigned Integer)
defm SSHLvvv : NeonI_3VSame_BHSD_sizes<0b0, 0b01000, "sshl",
                 int_arm_neon_vshifts, 1>;
defm USHLvvv : NeonI_3VSame_BHSD_sizes<0b1, 0b01000, "ushl",
                 int_arm_neon_vshiftu, 1>;

// Vector Saturating Shift Left (Signed and Unsigned Integer)
defm SQSHLvvv : NeonI_3VSame_BHSD_sizes<0b0, 0b01001, "sqshl",
                  int_arm_neon_vqshifts, 1>;
defm UQSHLvvv : NeonI_3VSame_BHSD_sizes<0b1, 0b01001, "uqshl",
                  int_arm_neon_vqshiftu, 1>;

// Vector Rouding Shift Left (Signed and Unsigned Integer)
defm SRSHLvvv : NeonI_3VSame_BHSD_sizes<0b0, 0b01010, "srshl",
                  int_arm_neon_vrshifts, 1>;
defm URSHLvvv : NeonI_3VSame_BHSD_sizes<0b1, 0b01010, "urshl",
                  int_arm_neon_vrshiftu, 1>;

// Vector Saturating Rouding Shift Left (Signed and Unsigned Integer)
defm SQRSHLvvv : NeonI_3VSame_BHSD_sizes<0b0, 0b01011, "sqrshl",
                   int_arm_neon_vqrshifts, 1>;
defm UQRSHLvvv : NeonI_3VSame_BHSD_sizes<0b1, 0b01011, "uqrshl",
                   int_arm_neon_vqrshiftu, 1>;

// Vector Maximum (Signed and Unsigned Integer)
defm SMAXvvv : NeonI_3VSame_BHS_sizes<0b0, 0b01100, "smax", int_arm_neon_vmaxs, 1>;
defm UMAXvvv : NeonI_3VSame_BHS_sizes<0b1, 0b01100, "umax", int_arm_neon_vmaxu, 1>;

// Vector Minimum (Signed and Unsigned Integer)
defm SMINvvv : NeonI_3VSame_BHS_sizes<0b0, 0b01101, "smin", int_arm_neon_vmins, 1>;
defm UMINvvv : NeonI_3VSame_BHS_sizes<0b1, 0b01101, "umin", int_arm_neon_vminu, 1>;

// Vector Maximum (Floating Point)
defm FMAXvvv : NeonI_3VSame_SD_sizes<0b0, 0b0, 0b11110, "fmax",
                                     int_arm_neon_vmaxs, int_arm_neon_vmaxs,
                                     int_arm_neon_vmaxs, v2f32, v4f32, v2f64, 1>;

// Vector Minimum (Floating Point)
defm FMINvvv : NeonI_3VSame_SD_sizes<0b0, 0b1, 0b11110, "fmin",
                                     int_arm_neon_vmins, int_arm_neon_vmins,
                                     int_arm_neon_vmins, v2f32, v4f32, v2f64, 1>;

// Vector maxNum (Floating Point) -  prefer a number over a quiet NaN)
defm FMAXNMvvv : NeonI_3VSame_SD_sizes<0b0, 0b0, 0b11000, "fmaxnm",
                                       int_aarch64_neon_vmaxnm,
                                       int_aarch64_neon_vmaxnm,
                                       int_aarch64_neon_vmaxnm,
                                       v2f32, v4f32, v2f64, 1>;

// Vector minNum (Floating Point) - prefer a number over a quiet NaN)
defm FMINNMvvv : NeonI_3VSame_SD_sizes<0b0, 0b1, 0b11000, "fminnm",
                                       int_aarch64_neon_vminnm,
                                       int_aarch64_neon_vminnm,
                                       int_aarch64_neon_vminnm,
                                       v2f32, v4f32, v2f64, 1>;

// Vector Maximum Pairwise (Signed and Unsigned Integer)
defm SMAXPvvv : NeonI_3VSame_BHS_sizes<0b0, 0b10100, "smaxp", int_arm_neon_vpmaxs, 1>;
defm UMAXPvvv : NeonI_3VSame_BHS_sizes<0b1, 0b10100, "umaxp", int_arm_neon_vpmaxu, 1>;

// Vector Minimum Pairwise (Signed and Unsigned Integer)
defm SMINPvvv : NeonI_3VSame_BHS_sizes<0b0, 0b10101, "sminp", int_arm_neon_vpmins, 1>;
defm UMINPvvv : NeonI_3VSame_BHS_sizes<0b1, 0b10101, "uminp", int_arm_neon_vpminu, 1>;

// Vector Maximum Pairwise (Floating Point)
defm FMAXPvvv : NeonI_3VSame_SD_sizes<0b1, 0b0, 0b11110, "fmaxp",
                                     int_arm_neon_vpmaxs, int_arm_neon_vpmaxs,
                                     int_arm_neon_vpmaxs, v2f32, v4f32, v2f64, 1>;

// Vector Minimum Pairwise (Floating Point)
defm FMINPvvv : NeonI_3VSame_SD_sizes<0b1, 0b1, 0b11110, "fminp",
                                     int_arm_neon_vpmins, int_arm_neon_vpmins,
                                     int_arm_neon_vpmins, v2f32, v4f32, v2f64, 1>;

// Vector maxNum Pairwise (Floating Point) -  prefer a number over a quiet NaN)
defm FMAXNMPvvv : NeonI_3VSame_SD_sizes<0b1, 0b0, 0b11000, "fmaxnmp",
                                       int_aarch64_neon_vpmaxnm,
                                       int_aarch64_neon_vpmaxnm,
                                       int_aarch64_neon_vpmaxnm,
                                       v2f32, v4f32, v2f64, 1>;

// Vector minNum Pairwise (Floating Point) -  prefer a number over a quiet NaN)
defm FMINNMPvvv : NeonI_3VSame_SD_sizes<0b1, 0b1, 0b11000, "fminnmp",
                                       int_aarch64_neon_vpminnm,
                                       int_aarch64_neon_vpminnm,
                                       int_aarch64_neon_vpminnm,
                                       v2f32, v4f32, v2f64, 1>;

// Vector Addition Pairwise (Integer)
defm ADDP : NeonI_3VSame_BHSD_sizes<0b0, 0b10111, "addp", int_arm_neon_vpadd, 1>;

// Vector Addition Pairwise (Floating Point)
defm FADDP : NeonI_3VSame_SD_sizes<0b1, 0b0, 0b11010, "faddp",
                                       int_arm_neon_vpadd,
                                       int_arm_neon_vpadd,
                                       int_arm_neon_vpadd,
                                       v2f32, v4f32, v2f64, 1>;

// Vector Saturating Doubling Multiply High
defm SQDMULHvvv : NeonI_3VSame_HS_sizes<0b0, 0b10110, "sqdmulh",
                    int_arm_neon_vqdmulh, 1>;

// Vector Saturating Rouding Doubling Multiply High
defm SQRDMULHvvv : NeonI_3VSame_HS_sizes<0b1, 0b10110, "sqrdmulh",
                     int_arm_neon_vqrdmulh, 1>;

// Vector Multiply Extended (Floating Point)
defm FMULXvvv : NeonI_3VSame_SD_sizes<0b0, 0b0, 0b11011, "fmulx",
                                      int_aarch64_neon_vmulx,
                                      int_aarch64_neon_vmulx,
                                      int_aarch64_neon_vmulx,
                                      v2f32, v4f32, v2f64, 1>;

// Vector Immediate Instructions

multiclass neon_mov_imm_shift_asmoperands<string PREFIX>
{
  def _asmoperand : AsmOperandClass
    {
      let Name = "NeonMovImmShift" # PREFIX;
      let RenderMethod = "addNeonMovImmShift" # PREFIX # "Operands";
      let PredicateMethod = "isNeonMovImmShift" # PREFIX;
    }
}

// Definition of vector immediates shift operands

// The selectable use-cases extract the shift operation
// information from the OpCmode fields encoded in the immediate.
def neon_mod_shift_imm_XFORM : SDNodeXForm<imm, [{
  uint64_t OpCmode = N->getZExtValue();
  unsigned ShiftImm;
  unsigned ShiftOnesIn;
  unsigned HasShift =
    A64Imms::decodeNeonModShiftImm(OpCmode, ShiftImm, ShiftOnesIn);
  if (!HasShift) return SDValue();
  return CurDAG->getTargetConstant(ShiftImm, MVT::i32);
}]>;

// Vector immediates shift operands which accept LSL and MSL
// shift operators with shift value in the range of 0, 8, 16, 24 (LSL),
// or 0, 8 (LSLH) or 8, 16 (MSL).
defm neon_mov_imm_LSL : neon_mov_imm_shift_asmoperands<"LSL">;
defm neon_mov_imm_MSL : neon_mov_imm_shift_asmoperands<"MSL">;
// LSLH restricts shift amount to  0, 8 out of 0, 8, 16, 24
defm neon_mov_imm_LSLH : neon_mov_imm_shift_asmoperands<"LSLH">;

multiclass neon_mov_imm_shift_operands<string PREFIX,
                                       string HALF, string ISHALF, code pred>
{
   def _operand : Operand<i32>, ImmLeaf<i32, pred, neon_mod_shift_imm_XFORM>
    {
      let PrintMethod =
        "printNeonMovImmShiftOperand<A64SE::" # PREFIX # ", " # ISHALF # ">";
      let DecoderMethod =
        "DecodeNeonMovImmShiftOperand<A64SE::" # PREFIX # ", " # ISHALF # ">";
      let ParserMatchClass =
        !cast<AsmOperandClass>("neon_mov_imm_" # PREFIX # HALF # "_asmoperand");
    }
}

defm neon_mov_imm_LSL  : neon_mov_imm_shift_operands<"LSL", "", "false", [{
  unsigned ShiftImm;
  unsigned ShiftOnesIn;
  unsigned HasShift =
    A64Imms::decodeNeonModShiftImm(Imm, ShiftImm, ShiftOnesIn);
  return (HasShift && !ShiftOnesIn);
}]>;

defm neon_mov_imm_MSL  : neon_mov_imm_shift_operands<"MSL", "", "false", [{
  unsigned ShiftImm;
  unsigned ShiftOnesIn;
  unsigned HasShift =
    A64Imms::decodeNeonModShiftImm(Imm, ShiftImm, ShiftOnesIn);
  return (HasShift && ShiftOnesIn);
}]>;

defm neon_mov_imm_LSLH  : neon_mov_imm_shift_operands<"LSL", "H", "true", [{
  unsigned ShiftImm;
  unsigned ShiftOnesIn;
  unsigned HasShift =
    A64Imms::decodeNeonModShiftImm(Imm, ShiftImm, ShiftOnesIn);
  return (HasShift && !ShiftOnesIn);
}]>;

def neon_uimm8_asmoperand : AsmOperandClass
{
  let Name = "UImm8";
  let PredicateMethod = "isUImm<8>";
  let RenderMethod = "addImmOperands";
}

def neon_uimm8 : Operand<i32>, ImmLeaf<i32, [{(void)Imm; return true;}]> {
  let ParserMatchClass = neon_uimm8_asmoperand;
  let PrintMethod = "printNeonUImm8Operand";
}

def neon_uimm64_mask_asmoperand : AsmOperandClass
{
  let Name = "NeonUImm64Mask";
  let PredicateMethod = "isNeonUImm64Mask";
  let RenderMethod = "addNeonUImm64MaskOperands";
}

// MCOperand for 64-bit bytemask with each byte having only the
// value 0x00 and 0xff is encoded as an unsigned 8-bit value
def neon_uimm64_mask : Operand<i32>, ImmLeaf<i32, [{(void)Imm; return true;}]> {
  let ParserMatchClass = neon_uimm64_mask_asmoperand;
  let PrintMethod = "printNeonUImm64MaskOperand";
}

multiclass NeonI_mov_imm_lsl_sizes<string asmop, bit op,
                                   SDPatternOperator opnode>
{
    // shift zeros, per word
    def _2S  : NeonI_1VModImm<0b0, op,
                              (outs VPR64:$Rd),
                              (ins neon_uimm8:$Imm,
                                neon_mov_imm_LSL_operand:$Simm),
                              !strconcat(asmop, " $Rd.2s, $Imm$Simm"),
                              [(set (v2i32 VPR64:$Rd),
                                 (v2i32 (opnode (timm:$Imm),
                                   (neon_mov_imm_LSL_operand:$Simm))))],
                              NoItinerary> {
       bits<2> Simm;
       let cmode = {0b0, Simm{1}, Simm{0}, 0b0};
     }

    def _4S  : NeonI_1VModImm<0b1, op,
                              (outs VPR128:$Rd),
                              (ins neon_uimm8:$Imm,
                                neon_mov_imm_LSL_operand:$Simm),
                              !strconcat(asmop, " $Rd.4s, $Imm$Simm"),
                              [(set (v4i32 VPR128:$Rd),
                                 (v4i32 (opnode (timm:$Imm),
                                   (neon_mov_imm_LSL_operand:$Simm))))],
                              NoItinerary> {
      bits<2> Simm;
      let cmode = {0b0, Simm{1}, Simm{0}, 0b0};
    }

    // shift zeros, per halfword
    def _4H  : NeonI_1VModImm<0b0, op,
                              (outs VPR64:$Rd),
                              (ins neon_uimm8:$Imm,
                                neon_mov_imm_LSLH_operand:$Simm),
                              !strconcat(asmop, " $Rd.4h, $Imm$Simm"),
                              [(set (v4i16 VPR64:$Rd),
                                 (v4i16 (opnode (timm:$Imm),
                                   (neon_mov_imm_LSLH_operand:$Simm))))],
                              NoItinerary> {
      bit  Simm;
      let cmode = {0b1, 0b0, Simm, 0b0};
    }

    def _8H  : NeonI_1VModImm<0b1, op,
                              (outs VPR128:$Rd),
                              (ins neon_uimm8:$Imm,
                                neon_mov_imm_LSLH_operand:$Simm),
                              !strconcat(asmop, " $Rd.8h, $Imm$Simm"),
                              [(set (v8i16 VPR128:$Rd),
                                 (v8i16 (opnode (timm:$Imm),
                                   (neon_mov_imm_LSLH_operand:$Simm))))],
                              NoItinerary> {
      bit Simm;
      let cmode = {0b1, 0b0, Simm, 0b0};
     }
}

multiclass NeonI_mov_imm_with_constraint_lsl_sizes<string asmop, bit op,
                                                   SDPatternOperator opnode,
                                                   SDPatternOperator neonopnode>
{
  let Constraints = "$src = $Rd" in {
    // shift zeros, per word
    def _2S  : NeonI_1VModImm<0b0, op,
                 (outs VPR64:$Rd),
                 (ins VPR64:$src, neon_uimm8:$Imm,
                   neon_mov_imm_LSL_operand:$Simm),
                 !strconcat(asmop, " $Rd.2s, $Imm$Simm"),
                 [(set (v2i32 VPR64:$Rd),
                    (v2i32 (opnode (v2i32 VPR64:$src),
                      (v2i32 (bitconvert (v2i32 (neonopnode timm:$Imm,
                        neon_mov_imm_LSL_operand:$Simm)))))))],
                 NoItinerary> {
      bits<2> Simm;
      let cmode = {0b0, Simm{1}, Simm{0}, 0b1};
    }

    def _4S  : NeonI_1VModImm<0b1, op,
                 (outs VPR128:$Rd),
                 (ins VPR128:$src, neon_uimm8:$Imm,
                   neon_mov_imm_LSL_operand:$Simm),
                 !strconcat(asmop, " $Rd.4s, $Imm$Simm"),
                 [(set (v4i32 VPR128:$Rd),
                    (v4i32 (opnode (v4i32 VPR128:$src),
                      (v4i32 (bitconvert (v4i32 (neonopnode timm:$Imm,
                        neon_mov_imm_LSL_operand:$Simm)))))))],
                 NoItinerary> {
      bits<2> Simm;
      let cmode = {0b0, Simm{1}, Simm{0}, 0b1};
    }

    // shift zeros, per halfword
    def _4H  : NeonI_1VModImm<0b0, op,
                 (outs VPR64:$Rd),
                 (ins VPR64:$src, neon_uimm8:$Imm,
                   neon_mov_imm_LSLH_operand:$Simm),
                 !strconcat(asmop, " $Rd.4h, $Imm$Simm"),
                 [(set (v4i16 VPR64:$Rd),
                    (v4i16 (opnode (v4i16 VPR64:$src),
                       (v4i16 (bitconvert (v4i16 (neonopnode timm:$Imm,
                          neon_mov_imm_LSL_operand:$Simm)))))))],
                 NoItinerary> {
      bit  Simm;
      let cmode = {0b1, 0b0, Simm, 0b1};
    }

    def _8H  : NeonI_1VModImm<0b1, op,
                 (outs VPR128:$Rd),
                 (ins VPR128:$src, neon_uimm8:$Imm,
                   neon_mov_imm_LSLH_operand:$Simm),
                 !strconcat(asmop, " $Rd.8h, $Imm$Simm"),
                 [(set (v8i16 VPR128:$Rd),
                    (v8i16 (opnode (v8i16 VPR128:$src),
                      (v8i16 (bitconvert (v8i16 (neonopnode timm:$Imm,
                        neon_mov_imm_LSL_operand:$Simm)))))))],
                 NoItinerary> {
      bit Simm;
      let cmode = {0b1, 0b0, Simm, 0b1};
    }
  }
}

multiclass NeonI_mov_imm_msl_sizes<string asmop, bit op,
                                   SDPatternOperator opnode>
{
    // shift ones, per word
    def _2S  : NeonI_1VModImm<0b0, op,
                             (outs VPR64:$Rd),
                             (ins neon_uimm8:$Imm,
                               neon_mov_imm_MSL_operand:$Simm),
                             !strconcat(asmop, " $Rd.2s, $Imm$Simm"),
                              [(set (v2i32 VPR64:$Rd),
                                 (v2i32 (opnode (timm:$Imm),
                                   (neon_mov_imm_MSL_operand:$Simm))))],
                             NoItinerary> {
       bit Simm;
       let cmode = {0b1, 0b1, 0b0, Simm};
     }

   def _4S  : NeonI_1VModImm<0b1, op,
                              (outs VPR128:$Rd),
                              (ins neon_uimm8:$Imm,
                                neon_mov_imm_MSL_operand:$Simm),
                              !strconcat(asmop, " $Rd.4s, $Imm$Simm"),
                              [(set (v4i32 VPR128:$Rd),
                                 (v4i32 (opnode (timm:$Imm),
                                   (neon_mov_imm_MSL_operand:$Simm))))],
                              NoItinerary> {
     bit Simm;
     let cmode = {0b1, 0b1, 0b0, Simm};
   }
}

// Vector Move Immediate Shifted
let isReMaterializable = 1 in {
defm MOVIvi_lsl : NeonI_mov_imm_lsl_sizes<"movi", 0b0, Neon_movi>;
}

// Vector Move Inverted Immediate Shifted
let isReMaterializable = 1 in {
defm MVNIvi_lsl : NeonI_mov_imm_lsl_sizes<"mvni", 0b1, Neon_mvni>;
}

// Vector Bitwise Bit Clear (AND NOT) - immediate
let isReMaterializable = 1 in {
defm BICvi_lsl : NeonI_mov_imm_with_constraint_lsl_sizes<"bic", 0b1,
                                                         and, Neon_mvni>;
}

// Vector Bitwise OR - immedidate

let isReMaterializable = 1 in {
defm ORRvi_lsl   : NeonI_mov_imm_with_constraint_lsl_sizes<"orr", 0b0,
                                                           or, Neon_movi>;
}

// Additional patterns for Vector Bitwise Bit Clear (AND NOT) - immedidate
// LowerBUILD_VECTOR favors lowering MOVI over MVNI.
// BIC immediate instructions selection requires additional patterns to
// transform Neon_movi operands into BIC immediate operands

def neon_mov_imm_LSLH_transform_XFORM : SDNodeXForm<imm, [{
  uint64_t OpCmode = N->getZExtValue();
  unsigned ShiftImm;
  unsigned ShiftOnesIn;
  (void)A64Imms::decodeNeonModShiftImm(OpCmode, ShiftImm, ShiftOnesIn);
  // LSLH restricts shift amount to  0, 8 which are encoded as 0 and 1
  // Transform encoded shift amount 0 to 1 and 1 to 0.
  return CurDAG->getTargetConstant(!ShiftImm, MVT::i32);
}]>;

def neon_mov_imm_LSLH_transform_operand
  : ImmLeaf<i32, [{
    unsigned ShiftImm;
    unsigned ShiftOnesIn;
    unsigned HasShift =
      A64Imms::decodeNeonModShiftImm(Imm, ShiftImm, ShiftOnesIn);
    return (HasShift && !ShiftOnesIn); }],
  neon_mov_imm_LSLH_transform_XFORM>;

// Transform (and A, (4h Neon_movi 0xff)) -> BIC 4h (A, 0x00, LSL 8)
// Transform (and A, (4h Neon_movi 0xff LSL #8)) -> BIC 4h (A, 0x00)
def : Pat<(v4i16 (and VPR64:$src,
            (v4i16 (Neon_movi 255, neon_mov_imm_LSLH_transform_operand:$Simm)))),
          (BICvi_lsl_4H VPR64:$src, 0,
            neon_mov_imm_LSLH_transform_operand:$Simm)>;

// Transform (and A, (8h Neon_movi 8h 0xff)) -> BIC 8h (A, 0x00, LSL 8)
// Transform (and A, (8h Neon_movi 0xff LSL #8)) -> BIC 8h (A, 0x00)
def : Pat<(v8i16 (and VPR128:$src,
            (v8i16 (Neon_movi 255, neon_mov_imm_LSLH_transform_operand:$Simm)))),
          (BICvi_lsl_8H VPR128:$src, 0,
            neon_mov_imm_LSLH_transform_operand:$Simm)>;


multiclass Neon_bitwiseVi_patterns<SDPatternOperator opnode,
                                   SDPatternOperator neonopnode,
                                   Instruction INST4H,
                                   Instruction INST8H> {
  def : Pat<(v8i8 (opnode VPR64:$src,
                    (bitconvert(v4i16 (neonopnode timm:$Imm,
                      neon_mov_imm_LSLH_operand:$Simm))))),
            (INST4H VPR64:$src, neon_uimm8:$Imm,
              neon_mov_imm_LSLH_operand:$Simm)>;
  def : Pat<(v1i64 (opnode VPR64:$src,
                  (bitconvert(v4i16 (neonopnode timm:$Imm,
                    neon_mov_imm_LSLH_operand:$Simm))))),
          (INST4H VPR64:$src, neon_uimm8:$Imm,
            neon_mov_imm_LSLH_operand:$Simm)>;

  def : Pat<(v16i8 (opnode VPR128:$src,
                   (bitconvert(v8i16 (neonopnode timm:$Imm,
                     neon_mov_imm_LSLH_operand:$Simm))))),
          (INST8H VPR128:$src, neon_uimm8:$Imm,
            neon_mov_imm_LSLH_operand:$Simm)>;
  def : Pat<(v4i32 (opnode VPR128:$src,
                   (bitconvert(v8i16 (neonopnode timm:$Imm,
                     neon_mov_imm_LSLH_operand:$Simm))))),
          (INST8H VPR128:$src, neon_uimm8:$Imm,
            neon_mov_imm_LSLH_operand:$Simm)>;
  def : Pat<(v2i64 (opnode VPR128:$src,
                   (bitconvert(v8i16 (neonopnode timm:$Imm,
                     neon_mov_imm_LSLH_operand:$Simm))))),
          (INST8H VPR128:$src, neon_uimm8:$Imm,
            neon_mov_imm_LSLH_operand:$Simm)>;
}

// Additional patterns for Vector Vector Bitwise Bit Clear (AND NOT) - immediate
defm : Neon_bitwiseVi_patterns<or, Neon_mvni, BICvi_lsl_4H, BICvi_lsl_8H>;

// Additional patterns for Vector Bitwise OR - immedidate
defm : Neon_bitwiseVi_patterns<or, Neon_movi, ORRvi_lsl_4H, ORRvi_lsl_8H>;


// Vector Move Immediate Masked
let isReMaterializable = 1 in {
defm MOVIvi_msl : NeonI_mov_imm_msl_sizes<"movi", 0b0, Neon_movi>;
}

// Vector Move Inverted Immediate Masked
let isReMaterializable = 1 in {
defm MVNIvi_msl : NeonI_mov_imm_msl_sizes<"mvni", 0b1, Neon_mvni>;
}

class NeonI_mov_imm_lsl_aliases<string asmop, string asmlane,
                                Instruction inst, RegisterClass VPRC>
  : NeonInstAlias<!strconcat(asmop, " $Rd," # asmlane # ", $Imm"),
                        (inst VPRC:$Rd, neon_uimm8:$Imm,  0), 0b0>;

// Aliases for Vector Move Immediate Shifted
def : NeonI_mov_imm_lsl_aliases<"movi", ".2s", MOVIvi_lsl_2S, VPR64>;
def : NeonI_mov_imm_lsl_aliases<"movi", ".4s", MOVIvi_lsl_4S, VPR128>;
def : NeonI_mov_imm_lsl_aliases<"movi", ".4h", MOVIvi_lsl_4H, VPR64>;
def : NeonI_mov_imm_lsl_aliases<"movi", ".8h", MOVIvi_lsl_8H, VPR128>;

// Aliases for Vector Move Inverted Immediate Shifted
def : NeonI_mov_imm_lsl_aliases<"mvni", ".2s", MVNIvi_lsl_2S, VPR64>;
def : NeonI_mov_imm_lsl_aliases<"mvni", ".4s", MVNIvi_lsl_4S, VPR128>;
def : NeonI_mov_imm_lsl_aliases<"mvni", ".4h", MVNIvi_lsl_4H, VPR64>;
def : NeonI_mov_imm_lsl_aliases<"mvni", ".8h", MVNIvi_lsl_8H, VPR128>;

// Aliases for Vector Bitwise Bit Clear (AND NOT) - immediate
def : NeonI_mov_imm_lsl_aliases<"bic", ".2s", BICvi_lsl_2S, VPR64>;
def : NeonI_mov_imm_lsl_aliases<"bic", ".4s", BICvi_lsl_4S, VPR128>;
def : NeonI_mov_imm_lsl_aliases<"bic", ".4h", BICvi_lsl_4H, VPR64>;
def : NeonI_mov_imm_lsl_aliases<"bic", ".8h", BICvi_lsl_8H, VPR128>;

// Aliases for Vector Bitwise OR - immedidate
def : NeonI_mov_imm_lsl_aliases<"orr", ".2s", ORRvi_lsl_2S, VPR64>;
def : NeonI_mov_imm_lsl_aliases<"orr", ".4s", ORRvi_lsl_4S, VPR128>;
def : NeonI_mov_imm_lsl_aliases<"orr", ".4h", ORRvi_lsl_4H, VPR64>;
def : NeonI_mov_imm_lsl_aliases<"orr", ".8h", ORRvi_lsl_8H, VPR128>;

//  Vector Move Immediate - per byte
let isReMaterializable = 1 in {
def MOVIvi_8B : NeonI_1VModImm<0b0, 0b0,
                               (outs VPR64:$Rd), (ins neon_uimm8:$Imm),
                               "movi\t$Rd.8b, $Imm",
                               [(set (v8i8 VPR64:$Rd),
                                  (v8i8 (Neon_movi (timm:$Imm), (i32 imm))))],
                                NoItinerary> {
  let cmode = 0b1110;
}

def MOVIvi_16B : NeonI_1VModImm<0b1, 0b0,
                                (outs VPR128:$Rd), (ins neon_uimm8:$Imm),
                                "movi\t$Rd.16b, $Imm",
                                [(set (v16i8 VPR128:$Rd),
                                   (v16i8 (Neon_movi (timm:$Imm), (i32 imm))))],
                                 NoItinerary> {
  let cmode = 0b1110;
}
}

// Vector Move Immediate - bytemask, per double word
let isReMaterializable = 1 in {
def MOVIvi_2D : NeonI_1VModImm<0b1, 0b1,
                               (outs VPR128:$Rd), (ins neon_uimm64_mask:$Imm),
                               "movi\t $Rd.2d, $Imm",
                               [(set (v2i64 VPR128:$Rd),
                                  (v2i64 (Neon_movi (timm:$Imm), (i32 imm))))],
                               NoItinerary> {
  let cmode = 0b1110;
}
}

// Vector Move Immediate - bytemask, one doubleword

let isReMaterializable = 1 in {
def MOVIdi : NeonI_1VModImm<0b0, 0b1,
                           (outs FPR64:$Rd), (ins neon_uimm64_mask:$Imm),
                           "movi\t $Rd, $Imm",
                           [(set (f64 FPR64:$Rd),
                              (f64 (bitconvert
                                (v1i64 (Neon_movi (timm:$Imm), (i32 imm))))))],
                           NoItinerary> {
  let cmode = 0b1110;
}
}

// Vector Floating Point Move Immediate

class NeonI_FMOV_impl<string asmlane, RegisterClass VPRC, ValueType OpTy,
                      Operand immOpType, bit q, bit op>
  : NeonI_1VModImm<q, op,
                   (outs VPRC:$Rd), (ins immOpType:$Imm),
                   "fmov\t$Rd" # asmlane # ", $Imm",
                   [(set (OpTy VPRC:$Rd),
                      (OpTy (Neon_fmovi (timm:$Imm))))],
                   NoItinerary> {
     let cmode = 0b1111;
   }

let isReMaterializable = 1 in {
def FMOVvi_2S : NeonI_FMOV_impl<".2s", VPR64,  v2f32, fmov32_operand, 0b0, 0b0>;
def FMOVvi_4S : NeonI_FMOV_impl<".4s", VPR128, v4f32, fmov32_operand, 0b1, 0b0>;
def FMOVvi_2D : NeonI_FMOV_impl<".2d", VPR128, v2f64, fmov64_operand, 0b1, 0b1>;
}

// Scalar Arithmetic

class NeonI_Scalar3Same_D_size<bit u, bits<5> opcode, string asmop>
  : NeonI_Scalar3Same<u, 0b11, opcode,
                (outs FPR64:$Rd), (ins FPR64:$Rn, FPR64:$Rm),
                !strconcat(asmop, " $Rd, $Rn, $Rm"),
                [],
                NoItinerary>;

multiclass NeonI_Scalar3Same_BHSD_sizes<bit u, bits<5> opcode,
                                        string asmop, bit Commutable = 0>
{
  let isCommutable = Commutable in {
    def bbb : NeonI_Scalar3Same<u, 0b00, opcode,
                                (outs FPR8:$Rd), (ins FPR8:$Rn, FPR8:$Rm),
                                !strconcat(asmop, " $Rd, $Rn, $Rm"),
                                [],
                                NoItinerary>;
    def hhh : NeonI_Scalar3Same<u, 0b01, opcode,
                                (outs FPR16:$Rd), (ins FPR16:$Rn, FPR16:$Rm),
                                !strconcat(asmop, " $Rd, $Rn, $Rm"),
                                [],
                                NoItinerary>;
    def sss : NeonI_Scalar3Same<u, 0b10, opcode,
                                (outs FPR32:$Rd), (ins FPR32:$Rn, FPR32:$Rm),
                                !strconcat(asmop, " $Rd, $Rn, $Rm"),
                                [],
                                NoItinerary>;
    def ddd : NeonI_Scalar3Same<u, 0b11, opcode,
                               (outs FPR64:$Rd), (ins FPR64:$Rn, FPR64:$Rm),
                               !strconcat(asmop, " $Rd, $Rn, $Rm"),
                               [],
                               NoItinerary>;
  }
}

class Neon_Scalar_D_size_patterns<SDPatternOperator opnode, Instruction INSTD>
  : Pat<(v1i64 (opnode (v1i64 VPR64:$Rn), (v1i64 VPR64:$Rm))),
        (SUBREG_TO_REG (i64 0),
              (INSTD (EXTRACT_SUBREG VPR64:$Rn, sub_64),
             (EXTRACT_SUBREG VPR64:$Rm, sub_64)),
          sub_64)>;


// Scalar Integer Add
let isCommutable = 1 in {
def ADDddd : NeonI_Scalar3Same_D_size<0b0, 0b10000, "add">;
}

// Scalar Integer Sub
def SUBddd : NeonI_Scalar3Same_D_size<0b1, 0b10000, "sub">;

// Pattern for Scalar Integer Add and Sub with D register
def : Neon_Scalar_D_size_patterns<add, ADDddd>;
def : Neon_Scalar_D_size_patterns<sub, SUBddd>;

// Scalar Integer Saturating Add (Signed, Unsigned)
defm SQADD : NeonI_Scalar3Same_BHSD_sizes<0b0, 0b00001, "sqadd", 1>;
defm UQADD : NeonI_Scalar3Same_BHSD_sizes<0b1, 0b00001, "uqadd", 1>;

// Scalar Integer Saturating Sub (Signed, Unsigned)
defm SQSUB : NeonI_Scalar3Same_BHSD_sizes<0b0, 0b00101, "sqsub", 0>;
defm UQSUB : NeonI_Scalar3Same_BHSD_sizes<0b1, 0b00101, "uqsub", 0>;

// Patterns for Scalar Integer Saturating Add, Sub with D register only
def : Neon_Scalar_D_size_patterns<int_arm_neon_vqadds, SQADDddd>;
def : Neon_Scalar_D_size_patterns<int_arm_neon_vqaddu, UQADDddd>;
def : Neon_Scalar_D_size_patterns<int_arm_neon_vqsubs, SQSUBddd>;
def : Neon_Scalar_D_size_patterns<int_arm_neon_vqsubu, UQSUBddd>;

// Scalar Integer Shift Left (Signed, Unsigned)
def SSHLddd : NeonI_Scalar3Same_D_size<0b0, 0b01000, "sshl">;
def USHLddd : NeonI_Scalar3Same_D_size<0b1, 0b01000, "ushl">;

// Scalar Integer Saturating Shift Left (Signed, Unsigned)
defm SQSHL: NeonI_Scalar3Same_BHSD_sizes<0b0, 0b01001, "sqshl", 0>;
defm UQSHL: NeonI_Scalar3Same_BHSD_sizes<0b1, 0b01001, "uqshl", 0>;

// Scalar Integer Rouding Shift Left (Signed, Unsigned)
def SRSHLddd: NeonI_Scalar3Same_D_size<0b0, 0b01010, "srshl">;
def URSHLddd: NeonI_Scalar3Same_D_size<0b1, 0b01010, "urshl">;

// Scalar Integer Saturating Rounding Shift Left (Signed, Unsigned)
defm SQRSHL: NeonI_Scalar3Same_BHSD_sizes<0b0, 0b01011, "sqrshl", 0>;
defm UQRSHL: NeonI_Scalar3Same_BHSD_sizes<0b1, 0b01011, "uqrshl", 0>;

// Patterns for Scalar Integer Shift Lef, Saturating Shift Left,
// Rounding Shift Left, Rounding Saturating Shift Left with D register only
def : Neon_Scalar_D_size_patterns<int_arm_neon_vshifts, SSHLddd>;
def : Neon_Scalar_D_size_patterns<int_arm_neon_vshiftu, USHLddd>;
def : Neon_Scalar_D_size_patterns<shl, SSHLddd>;
def : Neon_Scalar_D_size_patterns<shl, USHLddd>;
def : Neon_Scalar_D_size_patterns<int_arm_neon_vqshifts, SQSHLddd>;
def : Neon_Scalar_D_size_patterns<int_arm_neon_vqshiftu, UQSHLddd>;
def : Neon_Scalar_D_size_patterns<int_arm_neon_vrshifts, SRSHLddd>;
def : Neon_Scalar_D_size_patterns<int_arm_neon_vrshiftu, URSHLddd>;
def : Neon_Scalar_D_size_patterns<int_arm_neon_vqrshifts, SQRSHLddd>;
def : Neon_Scalar_D_size_patterns<int_arm_neon_vqrshiftu, UQRSHLddd>;


//===----------------------------------------------------------------------===//
// Non-Instruction Patterns
//===----------------------------------------------------------------------===//

// 64-bit vector bitcasts...

def : Pat<(v1i64 (bitconvert (v8i8  VPR64:$src))), (v1i64 VPR64:$src)>;
def : Pat<(v2f32 (bitconvert (v8i8  VPR64:$src))), (v2f32 VPR64:$src)>;
def : Pat<(v2i32 (bitconvert (v8i8  VPR64:$src))), (v2i32 VPR64:$src)>;
def : Pat<(v4i16 (bitconvert (v8i8  VPR64:$src))), (v4i16 VPR64:$src)>;

def : Pat<(v1i64 (bitconvert (v4i16  VPR64:$src))), (v1i64 VPR64:$src)>;
def : Pat<(v2i32 (bitconvert (v4i16  VPR64:$src))), (v2i32 VPR64:$src)>;
def : Pat<(v2f32 (bitconvert (v4i16  VPR64:$src))), (v2f32 VPR64:$src)>;
def : Pat<(v8i8  (bitconvert (v4i16  VPR64:$src))), (v8i8 VPR64:$src)>;

def : Pat<(v1i64 (bitconvert (v2i32  VPR64:$src))), (v1i64 VPR64:$src)>;
def : Pat<(v2f32 (bitconvert (v2i32  VPR64:$src))), (v2f32 VPR64:$src)>;
def : Pat<(v4i16 (bitconvert (v2i32  VPR64:$src))), (v4i16 VPR64:$src)>;
def : Pat<(v8i8  (bitconvert (v2i32  VPR64:$src))), (v8i8 VPR64:$src)>;

def : Pat<(v1i64 (bitconvert (v2f32  VPR64:$src))), (v1i64 VPR64:$src)>;
def : Pat<(v2i32 (bitconvert (v2f32  VPR64:$src))), (v2i32 VPR64:$src)>;
def : Pat<(v4i16 (bitconvert (v2f32  VPR64:$src))), (v4i16 VPR64:$src)>;
def : Pat<(v8i8  (bitconvert (v2f32  VPR64:$src))), (v8i8 VPR64:$src)>;

def : Pat<(v2f32 (bitconvert (v1i64  VPR64:$src))), (v2f32 VPR64:$src)>;
def : Pat<(v2i32 (bitconvert (v1i64  VPR64:$src))), (v2i32 VPR64:$src)>;
def : Pat<(v4i16 (bitconvert (v1i64  VPR64:$src))), (v4i16 VPR64:$src)>;
def : Pat<(v8i8  (bitconvert (v1i64  VPR64:$src))), (v8i8 VPR64:$src)>;

// ..and 128-bit vector bitcasts...

def : Pat<(v2f64 (bitconvert (v16i8  VPR128:$src))), (v2f64 VPR128:$src)>;
def : Pat<(v2i64 (bitconvert (v16i8  VPR128:$src))), (v2i64 VPR128:$src)>;
def : Pat<(v4f32 (bitconvert (v16i8  VPR128:$src))), (v4f32 VPR128:$src)>;
def : Pat<(v4i32 (bitconvert (v16i8  VPR128:$src))), (v4i32 VPR128:$src)>;
def : Pat<(v8i16 (bitconvert (v16i8  VPR128:$src))), (v8i16 VPR128:$src)>;

def : Pat<(v2f64 (bitconvert (v8i16  VPR128:$src))), (v2f64 VPR128:$src)>;
def : Pat<(v2i64 (bitconvert (v8i16  VPR128:$src))), (v2i64 VPR128:$src)>;
def : Pat<(v4i32 (bitconvert (v8i16  VPR128:$src))), (v4i32 VPR128:$src)>;
def : Pat<(v4f32 (bitconvert (v8i16  VPR128:$src))), (v4f32 VPR128:$src)>;
def : Pat<(v16i8 (bitconvert (v8i16  VPR128:$src))), (v16i8 VPR128:$src)>;

def : Pat<(v2f64 (bitconvert (v4i32  VPR128:$src))), (v2f64 VPR128:$src)>;
def : Pat<(v2i64 (bitconvert (v4i32  VPR128:$src))), (v2i64 VPR128:$src)>;
def : Pat<(v4f32 (bitconvert (v4i32  VPR128:$src))), (v4f32 VPR128:$src)>;
def : Pat<(v8i16 (bitconvert (v4i32  VPR128:$src))), (v8i16 VPR128:$src)>;
def : Pat<(v16i8 (bitconvert (v4i32  VPR128:$src))), (v16i8 VPR128:$src)>;

def : Pat<(v2f64 (bitconvert (v4f32  VPR128:$src))), (v2f64 VPR128:$src)>;
def : Pat<(v2i64 (bitconvert (v4f32  VPR128:$src))), (v2i64 VPR128:$src)>;
def : Pat<(v4i32 (bitconvert (v4f32  VPR128:$src))), (v4i32 VPR128:$src)>;
def : Pat<(v8i16 (bitconvert (v4f32  VPR128:$src))), (v8i16 VPR128:$src)>;
def : Pat<(v16i8 (bitconvert (v4f32  VPR128:$src))), (v16i8 VPR128:$src)>;

def : Pat<(v2f64 (bitconvert (v2i64  VPR128:$src))), (v2f64 VPR128:$src)>;
def : Pat<(v4f32 (bitconvert (v2i64  VPR128:$src))), (v4f32 VPR128:$src)>;
def : Pat<(v4i32 (bitconvert (v2i64  VPR128:$src))), (v4i32 VPR128:$src)>;
def : Pat<(v8i16 (bitconvert (v2i64  VPR128:$src))), (v8i16 VPR128:$src)>;
def : Pat<(v16i8 (bitconvert (v2i64  VPR128:$src))), (v16i8 VPR128:$src)>;

def : Pat<(v2i64 (bitconvert (v2f64  VPR128:$src))), (v2i64 VPR128:$src)>;
def : Pat<(v4f32 (bitconvert (v2f64  VPR128:$src))), (v4f32 VPR128:$src)>;
def : Pat<(v4i32 (bitconvert (v2f64  VPR128:$src))), (v4i32 VPR128:$src)>;
def : Pat<(v8i16 (bitconvert (v2f64  VPR128:$src))), (v8i16 VPR128:$src)>;
def : Pat<(v16i8 (bitconvert (v2f64  VPR128:$src))), (v16i8 VPR128:$src)>;


// ...and scalar bitcasts...

def : Pat<(f64   (bitconvert (v8i8  VPR64:$src))),
                 (f64 (EXTRACT_SUBREG (v8i8  VPR64:$src), sub_64))>;
def : Pat<(f64   (bitconvert (v4i16  VPR64:$src))),
                 (f64 (EXTRACT_SUBREG (v4i16  VPR64:$src), sub_64))>;
def : Pat<(f64   (bitconvert (v2i32  VPR64:$src))),
                 (f64 (EXTRACT_SUBREG (v2i32  VPR64:$src), sub_64))>;
def : Pat<(f64   (bitconvert (v2f32  VPR64:$src))),
                 (f64 (EXTRACT_SUBREG (v2f32  VPR64:$src), sub_64))>;
def : Pat<(f64   (bitconvert (v1i64  VPR64:$src))),
                 (f64 (EXTRACT_SUBREG (v1i64  VPR64:$src), sub_64))>;
def : Pat<(f128  (bitconvert (v16i8  VPR128:$src))),
                 (f128 (EXTRACT_SUBREG (v16i8  VPR128:$src), sub_alias))>;
def : Pat<(f128  (bitconvert (v8i16  VPR128:$src))),
                 (f128 (EXTRACT_SUBREG (v8i16  VPR128:$src), sub_alias))>;
def : Pat<(f128  (bitconvert (v4i32  VPR128:$src))),
                 (f128 (EXTRACT_SUBREG (v4i32  VPR128:$src), sub_alias))>;
def : Pat<(f128  (bitconvert (v2i64  VPR128:$src))),
                 (f128 (EXTRACT_SUBREG (v2i64  VPR128:$src), sub_alias))>;
def : Pat<(f128  (bitconvert (v4f32  VPR128:$src))),
                 (f128 (EXTRACT_SUBREG (v4f32  VPR128:$src), sub_alias))>;
def : Pat<(f128  (bitconvert (v2f64  VPR128:$src))),
                 (f128 (EXTRACT_SUBREG (v2f64  VPR128:$src), sub_alias))>;

def : Pat<(v8i8   (bitconvert (f64   FPR64:$src))),
                  (v8i8 (SUBREG_TO_REG (i64 0), (f64  FPR64:$src), sub_64))>;
def : Pat<(v4i16  (bitconvert (f64   FPR64:$src))),
                  (v4i16 (SUBREG_TO_REG (i64 0), (f64  FPR64:$src), sub_64))>;
def : Pat<(v2i32  (bitconvert (f64   FPR64:$src))),
                  (v2i32 (SUBREG_TO_REG (i64 0), (f64  FPR64:$src), sub_64))>;
def : Pat<(v2f32  (bitconvert (f64   FPR64:$src))),
                  (v2f32 (SUBREG_TO_REG (i64 0), (f64  FPR64:$src), sub_64))>;
def : Pat<(v1i64  (bitconvert (f64   FPR64:$src))),
                  (v1i64 (SUBREG_TO_REG (i64 0), (f64  FPR64:$src), sub_64))>;
def : Pat<(v16i8  (bitconvert (f128   FPR128:$src))),
                  (v16i8 (SUBREG_TO_REG (i128 0), (f128  FPR128:$src),
                  sub_alias))>;
def : Pat<(v8i16  (bitconvert (f128   FPR128:$src))),
                  (v8i16 (SUBREG_TO_REG (i128 0), (f128  FPR128:$src),
                  sub_alias))>;
def : Pat<(v4i32  (bitconvert (f128   FPR128:$src))),
                  (v4i32 (SUBREG_TO_REG (i128 0), (f128  FPR128:$src),
                  sub_alias))>;
def : Pat<(v2i64  (bitconvert (f128   FPR128:$src))),
                  (v2i64 (SUBREG_TO_REG (i128 0), (f128  FPR128:$src),
                  sub_alias))>;
def : Pat<(v4f32  (bitconvert (f128   FPR128:$src))),
                  (v4f32 (SUBREG_TO_REG (i128 0), (f128  FPR128:$src),
                  sub_alias))>;
def : Pat<(v2f64  (bitconvert (f128   FPR128:$src))),
                  (v2f64 (SUBREG_TO_REG (i128 0), (f128  FPR128:$src),
                  sub_alias))>;
