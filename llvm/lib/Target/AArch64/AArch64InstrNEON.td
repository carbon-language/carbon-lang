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

def SDTARMVSH : SDTypeProfile<1, 2, [SDTCisVec<0>, SDTCisSameAs<0, 1>,
                                     SDTCisVT<2, i32>]>;
def Neon_sqrshlImm   : SDNode<"AArch64ISD::NEON_QSHLs", SDTARMVSH>;
def Neon_uqrshlImm   : SDNode<"AArch64ISD::NEON_QSHLu", SDTARMVSH>;

def Neon_vdup : SDNode<"AArch64ISD::NEON_VDUP", SDTypeProfile<1, 1,
                       [SDTCisVec<0>]>>;
def Neon_vduplane : SDNode<"AArch64ISD::NEON_VDUPLANE", SDTypeProfile<1, 2,
                           [SDTCisVec<0>, SDTCisVec<1>, SDTCisVT<2, i64>]>>;

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
  RegisterOperand VPRC, ValueType OpTy, bit q, bit u, bits<2> size, 
  bits<5> opcode, SDPatternOperator opnode>
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
// Alias for ORR if Vn=Vm.
// FIXME: This is actually the preferred syntax but TableGen can't deal with
// custom printing of aliases.
def : NeonInstAlias<"mov $Rd.8b, $Rn.8b",
                    (ORRvvv_8B VPR64:$Rd, VPR64:$Rn, VPR64:$Rn), 0>;
def : NeonInstAlias<"mov $Rd.16b, $Rn.16b",
                    (ORRvvv_16B VPR128:$Rd, VPR128:$Rn, VPR128:$Rn), 0>;

def Neon_immAllOnes: PatLeaf<(Neon_movi (i32 timm), (i32 imm)), [{
  ConstantSDNode *ImmConstVal = cast<ConstantSDNode>(N->getOperand(0));
  ConstantSDNode *OpCmodeConstVal = cast<ConstantSDNode>(N->getOperand(1));
  unsigned EltBits;
  uint64_t EltVal = A64Imms::decodeNeonModImm(ImmConstVal->getZExtValue(),
    OpCmodeConstVal->getZExtValue(), EltBits);
  return (EltBits == 8 && EltVal == 0xff);
}]>;

def Neon_immAllZeros: PatLeaf<(Neon_movi (i32 timm), (i32 imm)), [{
  ConstantSDNode *ImmConstVal = cast<ConstantSDNode>(N->getOperand(0));
  ConstantSDNode *OpCmodeConstVal = cast<ConstantSDNode>(N->getOperand(1));
  unsigned EltBits;
  uint64_t EltVal = A64Imms::decodeNeonModImm(ImmConstVal->getZExtValue(),
    OpCmodeConstVal->getZExtValue(), EltBits);
  return (EltBits == 8 && EltVal == 0x0);
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
                            Instruction inst, RegisterOperand VPRC>
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

def neon_uimm1_asmoperand : AsmOperandClass
{
  let Name = "UImm1";
  let PredicateMethod = "isUImm<1>";
  let RenderMethod = "addImmOperands";
}

def neon_uimm2_asmoperand : AsmOperandClass
{
  let Name = "UImm2";
  let PredicateMethod = "isUImm<2>";
  let RenderMethod = "addImmOperands";
}

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
                                Instruction inst, RegisterOperand VPRC>
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

class NeonI_FMOV_impl<string asmlane, RegisterOperand VPRC, ValueType OpTy,
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

// Vector Shift (Immediate) 
// Immediate in [0, 63]
def imm0_63 : Operand<i32> {
  let ParserMatchClass = uimm6_asmoperand;
}

// Shift Right Immediate - A shift right immediate is encoded differently from
// other shift immediates. The immh:immb field is encoded like so:
//
//    Offset    Encoding
//     8        immh:immb<6:3> = '0001xxx', <imm> is encoded in immh:immb<2:0>
//     16       immh:immb<6:4> = '001xxxx', <imm> is encoded in immh:immb<3:0>
//     32       immh:immb<6:5> = '01xxxxx', <imm> is encoded in immh:immb<4:0>
//     64       immh:immb<6>   = '1xxxxxx', <imm> is encoded in immh:immb<5:0>
class shr_imm_asmoperands<string OFFSET> : AsmOperandClass {
  let Name = "ShrImm" # OFFSET;
  let RenderMethod = "addImmOperands";
  let DiagnosticType = "ShrImm" # OFFSET;
}

class shr_imm<string OFFSET> : Operand<i32> {
  let EncoderMethod = "getShiftRightImm" # OFFSET;
  let DecoderMethod = "DecodeShiftRightImm" # OFFSET;
  let ParserMatchClass = 
    !cast<AsmOperandClass>("shr_imm" # OFFSET # "_asmoperand");
}

def shr_imm8_asmoperand : shr_imm_asmoperands<"8">;
def shr_imm16_asmoperand : shr_imm_asmoperands<"16">;
def shr_imm32_asmoperand : shr_imm_asmoperands<"32">;
def shr_imm64_asmoperand : shr_imm_asmoperands<"64">;

def shr_imm8 : shr_imm<"8">;
def shr_imm16 : shr_imm<"16">;
def shr_imm32 : shr_imm<"32">;
def shr_imm64 : shr_imm<"64">;

class N2VShift<bit q, bit u, bits<5> opcode, string asmop, string T,
               RegisterOperand VPRC, ValueType Ty, Operand ImmTy, SDNode OpNode>
  : NeonI_2VShiftImm<q, u, opcode,
                     (outs VPRC:$Rd), (ins VPRC:$Rn, ImmTy:$Imm),
                     asmop # "\t$Rd." # T # ", $Rn." # T # ", $Imm",
                     [(set (Ty VPRC:$Rd),
                        (Ty (OpNode (Ty VPRC:$Rn),
                          (Ty (Neon_vdup (i32 imm:$Imm))))))],
                     NoItinerary>;

multiclass NeonI_N2VShL<bit u, bits<5> opcode, string asmop> {
  // 64-bit vector types.
  def _8B : N2VShift<0b0, u, opcode, asmop, "8b", VPR64, v8i8, uimm3, shl> {
    let Inst{22-19} = 0b0001;  // immh:immb = 0001xxx
  }

  def _4H : N2VShift<0b0, u, opcode, asmop, "4h", VPR64, v4i16, uimm4, shl> {
    let Inst{22-20} = 0b001;   // immh:immb = 001xxxx
  }

  def _2S : N2VShift<0b0, u, opcode, asmop, "2s", VPR64, v2i32, uimm5, shl> {
    let Inst{22-21} = 0b01;    // immh:immb = 01xxxxx
  }

  // 128-bit vector types.
  def _16B : N2VShift<0b1, u, opcode, asmop, "16b", VPR128, v16i8, uimm3, shl> {
    let Inst{22-19} = 0b0001;  // immh:immb = 0001xxx
  }

  def _8H : N2VShift<0b1, u, opcode, asmop, "8h", VPR128, v8i16, uimm4, shl> {
    let Inst{22-20} = 0b001;   // immh:immb = 001xxxx
  }

  def _4S : N2VShift<0b1, u, opcode, asmop, "4s", VPR128, v4i32, uimm5, shl> {
    let Inst{22-21} = 0b01;    // immh:immb = 01xxxxx
  }

  def _2D : N2VShift<0b1, u, opcode, asmop, "2d", VPR128, v2i64, imm0_63, shl> {
    let Inst{22} = 0b1;        // immh:immb = 1xxxxxx
  }
}

multiclass NeonI_N2VShR<bit u, bits<5> opcode, string asmop, SDNode OpNode> {
  def _8B : N2VShift<0b0, u, opcode, asmop, "8b", VPR64, v8i8, shr_imm8,
                     OpNode> {
    let Inst{22-19} = 0b0001;
  }

  def _4H : N2VShift<0b0, u, opcode, asmop, "4h", VPR64, v4i16, shr_imm16,
                     OpNode> {
    let Inst{22-20} = 0b001;
  }

  def _2S : N2VShift<0b0, u, opcode, asmop, "2s", VPR64, v2i32, shr_imm32,
                     OpNode> {
     let Inst{22-21} = 0b01;
  }

  def _16B : N2VShift<0b1, u, opcode, asmop, "16b", VPR128, v16i8, shr_imm8,
                      OpNode> {
                      let Inst{22-19} = 0b0001;
                    }

  def _8H : N2VShift<0b1, u, opcode, asmop, "8h", VPR128, v8i16, shr_imm16,
                     OpNode> {
                     let Inst{22-20} = 0b001;
                    }

  def _4S : N2VShift<0b1, u, opcode, asmop, "4s", VPR128, v4i32, shr_imm32,
                     OpNode> {
                      let Inst{22-21} = 0b01;
                    }

  def _2D : N2VShift<0b1, u, opcode, asmop, "2d", VPR128, v2i64, shr_imm64,
                     OpNode> {
                      let Inst{22} = 0b1;
                    }
}

// Shift left
defm SHLvvi : NeonI_N2VShL<0b0, 0b01010, "shl">;

// Shift right
defm SSHRvvi : NeonI_N2VShR<0b0, 0b00000, "sshr", sra>;
defm USHRvvi : NeonI_N2VShR<0b1, 0b00000, "ushr", srl>;

def Neon_High16B : PatFrag<(ops node:$in),
                           (extract_subvector (v16i8 node:$in), (iPTR 8))>;
def Neon_High8H  : PatFrag<(ops node:$in),
                           (extract_subvector (v8i16 node:$in), (iPTR 4))>;
def Neon_High4S  : PatFrag<(ops node:$in),
                           (extract_subvector (v4i32 node:$in), (iPTR 2))>;

def Neon_low8H : PatFrag<(ops node:$in),
                         (v4i16 (extract_subvector (v8i16 node:$in),
                                                   (iPTR 0)))>;
def Neon_low4S : PatFrag<(ops node:$in),
                         (v2i32 (extract_subvector (v4i32 node:$in),
                                                   (iPTR 0)))>;
def Neon_low4f : PatFrag<(ops node:$in),
                         (v2f32 (extract_subvector (v4f32 node:$in),
                                                   (iPTR 0)))>;

class N2VShiftLong<bit q, bit u, bits<5> opcode, string asmop, string DestT,
                   string SrcT, ValueType DestTy, ValueType SrcTy,
                   Operand ImmTy, SDPatternOperator ExtOp>
  : NeonI_2VShiftImm<q, u, opcode, (outs VPR128:$Rd),
                     (ins VPR64:$Rn, ImmTy:$Imm),
                     asmop # "\t$Rd." # DestT # ", $Rn." # SrcT # ", $Imm",
                     [(set (DestTy VPR128:$Rd),
                        (DestTy (shl
                          (DestTy (ExtOp (SrcTy VPR64:$Rn))),
                            (DestTy (Neon_vdup (i32 imm:$Imm))))))],
                     NoItinerary>;

class N2VShiftLongHigh<bit q, bit u, bits<5> opcode, string asmop, string DestT,
                       string SrcT, ValueType DestTy, ValueType SrcTy,
                       int StartIndex, Operand ImmTy,
                       SDPatternOperator ExtOp, PatFrag getTop>
  : NeonI_2VShiftImm<q, u, opcode, (outs VPR128:$Rd),
                     (ins VPR128:$Rn, ImmTy:$Imm),
                     asmop # "2\t$Rd." # DestT # ", $Rn." # SrcT # ", $Imm",
                     [(set (DestTy VPR128:$Rd),
                        (DestTy (shl
                          (DestTy (ExtOp
                            (SrcTy (getTop VPR128:$Rn)))),
                              (DestTy (Neon_vdup (i32 imm:$Imm))))))],
                     NoItinerary>;

multiclass NeonI_N2VShLL<string prefix, bit u, bits<5> opcode, string asmop,
                         SDNode ExtOp> {
  // 64-bit vector types.
  def _8B : N2VShiftLong<0b0, u, opcode, asmop, "8h", "8b", v8i16, v8i8,
                         uimm3, ExtOp> {
    let Inst{22-19} = 0b0001;  // immh:immb = 0001xxx
  }

  def _4H : N2VShiftLong<0b0, u, opcode, asmop, "4s", "4h", v4i32, v4i16,
                         uimm4, ExtOp> {
    let Inst{22-20} = 0b001;   // immh:immb = 001xxxx
  }

  def _2S : N2VShiftLong<0b0, u, opcode, asmop, "2d", "2s", v2i64, v2i32,
                         uimm5, ExtOp> {
    let Inst{22-21} = 0b01;    // immh:immb = 01xxxxx
  }

  // 128-bit vector types
  def _16B : N2VShiftLongHigh<0b1, u, opcode, asmop, "8h", "16b",
                              v8i16, v8i8, 8, uimm3, ExtOp, Neon_High16B> {
    let Inst{22-19} = 0b0001;  // immh:immb = 0001xxx
  }

  def _8H : N2VShiftLongHigh<0b1, u, opcode, asmop, "4s", "8h",
                             v4i32, v4i16, 4, uimm4, ExtOp, Neon_High8H> {
    let Inst{22-20} = 0b001;   // immh:immb = 001xxxx
  }

  def _4S : N2VShiftLongHigh<0b1, u, opcode, asmop, "2d", "4s",
                             v2i64, v2i32, 2, uimm5, ExtOp, Neon_High4S> {
    let Inst{22-21} = 0b01;    // immh:immb = 01xxxxx
  }

  // Use other patterns to match when the immediate is 0.
  def : Pat<(v8i16 (ExtOp (v8i8 VPR64:$Rn))),
            (!cast<Instruction>(prefix # "_8B") VPR64:$Rn, 0)>;

  def : Pat<(v4i32 (ExtOp (v4i16 VPR64:$Rn))),
            (!cast<Instruction>(prefix # "_4H") VPR64:$Rn, 0)>;

  def : Pat<(v2i64 (ExtOp (v2i32 VPR64:$Rn))),
            (!cast<Instruction>(prefix # "_2S") VPR64:$Rn, 0)>;

  def : Pat<(v8i16 (ExtOp (v8i8 (Neon_High16B VPR128:$Rn)))),
            (!cast<Instruction>(prefix # "_16B") VPR128:$Rn, 0)>;

  def : Pat<(v4i32 (ExtOp (v4i16 (Neon_High8H VPR128:$Rn)))),
            (!cast<Instruction>(prefix # "_8H") VPR128:$Rn, 0)>;

  def : Pat<(v2i64 (ExtOp (v2i32 (Neon_High4S VPR128:$Rn)))),
            (!cast<Instruction>(prefix # "_4S") VPR128:$Rn, 0)>;
}

// Shift left long
defm SSHLLvvi : NeonI_N2VShLL<"SSHLLvvi", 0b0, 0b10100, "sshll", sext>;
defm USHLLvvi : NeonI_N2VShLL<"USHLLvvi", 0b1, 0b10100, "ushll", zext>;

// Rounding/Saturating shift
class N2VShift_RQ<bit q, bit u, bits<5> opcode, string asmop, string T,
                  RegisterOperand VPRC, ValueType Ty, Operand ImmTy,
                  SDPatternOperator OpNode>
  : NeonI_2VShiftImm<q, u, opcode,
                     (outs VPRC:$Rd), (ins VPRC:$Rn, ImmTy:$Imm),
                     asmop # "\t$Rd." # T # ", $Rn." # T # ", $Imm",
                     [(set (Ty VPRC:$Rd), (Ty (OpNode (Ty VPRC:$Rn),
                        (i32 imm:$Imm))))],
                     NoItinerary>;

// shift right (vector by immediate)
multiclass NeonI_N2VShR_RQ<bit u, bits<5> opcode, string asmop,
                           SDPatternOperator OpNode> {
  def _8B  : N2VShift_RQ<0b0, u, opcode, asmop, "8b", VPR64, v8i8, shr_imm8,
                         OpNode> {
    let Inst{22-19} = 0b0001;
  }

  def _4H  : N2VShift_RQ<0b0, u, opcode, asmop, "4h", VPR64, v4i16, shr_imm16,
                         OpNode> {
    let Inst{22-20} = 0b001;
  }

  def _2S  : N2VShift_RQ<0b0, u, opcode, asmop, "2s", VPR64, v2i32, shr_imm32,
                         OpNode> {
    let Inst{22-21} = 0b01;
  }

  def _16B : N2VShift_RQ<0b1, u, opcode, asmop, "16b", VPR128, v16i8, shr_imm8,
                         OpNode> {
    let Inst{22-19} = 0b0001;
  }

  def _8H : N2VShift_RQ<0b1, u, opcode, asmop, "8h", VPR128, v8i16, shr_imm16,
                        OpNode> {
    let Inst{22-20} = 0b001;
  }

  def _4S : N2VShift_RQ<0b1, u, opcode, asmop, "4s", VPR128, v4i32, shr_imm32,
                        OpNode> {
    let Inst{22-21} = 0b01;
  }

  def _2D : N2VShift_RQ<0b1, u, opcode, asmop, "2d", VPR128, v2i64, shr_imm64,
                        OpNode> {
    let Inst{22} = 0b1;
  }
}

multiclass NeonI_N2VShL_Q<bit u, bits<5> opcode, string asmop,
                          SDPatternOperator OpNode> {
  // 64-bit vector types.
  def _8B : N2VShift_RQ<0b0, u, opcode, asmop, "8b", VPR64, v8i8, uimm3,
                        OpNode> {
    let Inst{22-19} = 0b0001;
  }

  def _4H : N2VShift_RQ<0b0, u, opcode, asmop, "4h", VPR64, v4i16, uimm4,
                        OpNode> {
    let Inst{22-20} = 0b001;
  }

  def _2S : N2VShift_RQ<0b0, u, opcode, asmop, "2s", VPR64, v2i32, uimm5,
                        OpNode> {
    let Inst{22-21} = 0b01;
  }

  // 128-bit vector types.
  def _16B : N2VShift_RQ<0b1, u, opcode, asmop, "16b", VPR128, v16i8, uimm3,
                         OpNode> {
    let Inst{22-19} = 0b0001;
  }

  def _8H : N2VShift_RQ<0b1, u, opcode, asmop, "8h", VPR128, v8i16, uimm4,
                        OpNode> {
    let Inst{22-20} = 0b001;
  }

  def _4S : N2VShift_RQ<0b1, u, opcode, asmop, "4s", VPR128, v4i32, uimm5,
                        OpNode> {
    let Inst{22-21} = 0b01;
  }

  def _2D : N2VShift_RQ<0b1, u, opcode, asmop, "2d", VPR128, v2i64, imm0_63,
                        OpNode> {
    let Inst{22} = 0b1;
  }
}

// Rounding shift right
defm SRSHRvvi : NeonI_N2VShR_RQ<0b0, 0b00100, "srshr",
                                int_aarch64_neon_vsrshr>;
defm URSHRvvi : NeonI_N2VShR_RQ<0b1, 0b00100, "urshr",
                                int_aarch64_neon_vurshr>;

// Saturating shift left unsigned
defm SQSHLUvvi : NeonI_N2VShL_Q<0b1, 0b01100, "sqshlu", int_aarch64_neon_vsqshlu>;

// Saturating shift left
defm SQSHLvvi : NeonI_N2VShL_Q<0b0, 0b01110, "sqshl", Neon_sqrshlImm>;
defm UQSHLvvi : NeonI_N2VShL_Q<0b1, 0b01110, "uqshl", Neon_uqrshlImm>;

class N2VShiftAdd<bit q, bit u, bits<5> opcode, string asmop, string T,
                  RegisterOperand VPRC, ValueType Ty, Operand ImmTy,
                  SDNode OpNode>
  : NeonI_2VShiftImm<q, u, opcode,
           (outs VPRC:$Rd), (ins VPRC:$src, VPRC:$Rn, ImmTy:$Imm),
           asmop # "\t$Rd." # T # ", $Rn." # T # ", $Imm",
           [(set (Ty VPRC:$Rd), (Ty (add (Ty VPRC:$src),
              (Ty (OpNode (Ty VPRC:$Rn),
                (Ty (Neon_vdup (i32 imm:$Imm))))))))],
           NoItinerary> {
  let Constraints = "$src = $Rd";
}

// Shift Right accumulate
multiclass NeonI_N2VShRAdd<bit u, bits<5> opcode, string asmop, SDNode OpNode> {
  def _8B : N2VShiftAdd<0b0, u, opcode, asmop, "8b", VPR64, v8i8, shr_imm8,
                        OpNode> {
    let Inst{22-19} = 0b0001;
  }

  def _4H : N2VShiftAdd<0b0, u, opcode, asmop, "4h", VPR64, v4i16, shr_imm16,
                        OpNode> {
    let Inst{22-20} = 0b001;
  }

  def _2S : N2VShiftAdd<0b0, u, opcode, asmop, "2s", VPR64, v2i32, shr_imm32,
                        OpNode> {
    let Inst{22-21} = 0b01;
  }

  def _16B : N2VShiftAdd<0b1, u, opcode, asmop, "16b", VPR128, v16i8, shr_imm8,
                         OpNode> {
    let Inst{22-19} = 0b0001;
  }

  def _8H : N2VShiftAdd<0b1, u, opcode, asmop, "8h", VPR128, v8i16, shr_imm16,
                        OpNode> {
    let Inst{22-20} = 0b001;
  }

  def _4S : N2VShiftAdd<0b1, u, opcode, asmop, "4s", VPR128, v4i32, shr_imm32,
                        OpNode> {
    let Inst{22-21} = 0b01;
  }

  def _2D : N2VShiftAdd<0b1, u, opcode, asmop, "2d", VPR128, v2i64, shr_imm64,
                        OpNode> {
    let Inst{22} = 0b1;
  }
}

// Shift right and accumulate
defm SSRAvvi    : NeonI_N2VShRAdd<0, 0b00010, "ssra", sra>;
defm USRAvvi    : NeonI_N2VShRAdd<1, 0b00010, "usra", srl>;

// Rounding shift accumulate
class N2VShiftAdd_R<bit q, bit u, bits<5> opcode, string asmop, string T,
                    RegisterOperand VPRC, ValueType Ty, Operand ImmTy,
                    SDPatternOperator OpNode>
  : NeonI_2VShiftImm<q, u, opcode,
                     (outs VPRC:$Rd), (ins VPRC:$src, VPRC:$Rn, ImmTy:$Imm),
                     asmop # "\t$Rd." # T # ", $Rn." # T # ", $Imm",
                     [(set (Ty VPRC:$Rd), (Ty (add (Ty VPRC:$src),
                        (Ty (OpNode (Ty VPRC:$Rn), (i32 imm:$Imm))))))],
                     NoItinerary> {
  let Constraints = "$src = $Rd";
}

multiclass NeonI_N2VShRAdd_R<bit u, bits<5> opcode, string asmop,
                             SDPatternOperator OpNode> {
  def _8B : N2VShiftAdd_R<0b0, u, opcode, asmop, "8b", VPR64, v8i8, shr_imm8,
                          OpNode> {
    let Inst{22-19} = 0b0001;
  }

  def _4H : N2VShiftAdd_R<0b0, u, opcode, asmop, "4h", VPR64, v4i16, shr_imm16,
                          OpNode> {
    let Inst{22-20} = 0b001;
  }

  def _2S : N2VShiftAdd_R<0b0, u, opcode, asmop, "2s", VPR64, v2i32, shr_imm32,
                          OpNode> {
    let Inst{22-21} = 0b01;
  }

  def _16B : N2VShiftAdd_R<0b1, u, opcode, asmop, "16b", VPR128, v16i8, shr_imm8,
                           OpNode> {
    let Inst{22-19} = 0b0001;
  }

  def _8H : N2VShiftAdd_R<0b1, u, opcode, asmop, "8h", VPR128, v8i16, shr_imm16,
                          OpNode> {
    let Inst{22-20} = 0b001;
  }

  def _4S : N2VShiftAdd_R<0b1, u, opcode, asmop, "4s", VPR128, v4i32, shr_imm32,
                          OpNode> {
    let Inst{22-21} = 0b01;
  }

  def _2D : N2VShiftAdd_R<0b1, u, opcode, asmop, "2d", VPR128, v2i64, shr_imm64,
                          OpNode> {
    let Inst{22} = 0b1;
  }
}

// Rounding shift right and accumulate
defm SRSRAvvi : NeonI_N2VShRAdd_R<0, 0b00110, "srsra", int_aarch64_neon_vsrshr>;
defm URSRAvvi : NeonI_N2VShRAdd_R<1, 0b00110, "ursra", int_aarch64_neon_vurshr>;

// Shift insert by immediate
class N2VShiftIns<bit q, bit u, bits<5> opcode, string asmop, string T,
                  RegisterOperand VPRC, ValueType Ty, Operand ImmTy,
                  SDPatternOperator OpNode>
    : NeonI_2VShiftImm<q, u, opcode,
           (outs VPRC:$Rd), (ins VPRC:$src, VPRC:$Rn, ImmTy:$Imm),
           asmop # "\t$Rd." # T # ", $Rn." # T # ", $Imm",
           [(set (Ty VPRC:$Rd), (Ty (OpNode (Ty VPRC:$src), (Ty VPRC:$Rn),
             (i32 imm:$Imm))))],
           NoItinerary> {
  let Constraints = "$src = $Rd";
}

// shift left insert (vector by immediate)
multiclass NeonI_N2VShLIns<bit u, bits<5> opcode, string asmop> {
  def _8B : N2VShiftIns<0b0, u, opcode, asmop, "8b", VPR64, v8i8, uimm3,
                        int_aarch64_neon_vsli> {
    let Inst{22-19} = 0b0001;
  }

  def _4H : N2VShiftIns<0b0, u, opcode, asmop, "4h", VPR64, v4i16, uimm4,
                        int_aarch64_neon_vsli> {
    let Inst{22-20} = 0b001;
  }

  def _2S : N2VShiftIns<0b0, u, opcode, asmop, "2s", VPR64, v2i32, uimm5,
                        int_aarch64_neon_vsli> {
    let Inst{22-21} = 0b01;
  }

    // 128-bit vector types
  def _16B : N2VShiftIns<0b1, u, opcode, asmop, "16b", VPR128, v16i8, uimm3,
                         int_aarch64_neon_vsli> {
    let Inst{22-19} = 0b0001;
  }

  def _8H : N2VShiftIns<0b1, u, opcode, asmop, "8h", VPR128, v8i16, uimm4,
                        int_aarch64_neon_vsli> {
    let Inst{22-20} = 0b001;
  }

  def _4S : N2VShiftIns<0b1, u, opcode, asmop, "4s", VPR128, v4i32, uimm5,
                        int_aarch64_neon_vsli> {
    let Inst{22-21} = 0b01;
  }

  def _2D : N2VShiftIns<0b1, u, opcode, asmop, "2d", VPR128, v2i64, imm0_63,
                        int_aarch64_neon_vsli> {
    let Inst{22} = 0b1;
  }
}

// shift right insert (vector by immediate)
multiclass NeonI_N2VShRIns<bit u, bits<5> opcode, string asmop> {
    // 64-bit vector types.
  def _8B : N2VShiftIns<0b0, u, opcode, asmop, "8b", VPR64, v8i8, shr_imm8,
                        int_aarch64_neon_vsri> {
    let Inst{22-19} = 0b0001;
  }

  def _4H : N2VShiftIns<0b0, u, opcode, asmop, "4h", VPR64, v4i16, shr_imm16,
                        int_aarch64_neon_vsri> {
    let Inst{22-20} = 0b001;
  }

  def _2S : N2VShiftIns<0b0, u, opcode, asmop, "2s", VPR64, v2i32, shr_imm32,
                        int_aarch64_neon_vsri> {
    let Inst{22-21} = 0b01;
  }

    // 128-bit vector types
  def _16B : N2VShiftIns<0b1, u, opcode, asmop, "16b", VPR128, v16i8, shr_imm8,
                         int_aarch64_neon_vsri> {
    let Inst{22-19} = 0b0001;
  }

  def _8H : N2VShiftIns<0b1, u, opcode, asmop, "8h", VPR128, v8i16, shr_imm16,
                        int_aarch64_neon_vsri> {
    let Inst{22-20} = 0b001;
  }

  def _4S : N2VShiftIns<0b1, u, opcode, asmop, "4s", VPR128, v4i32, shr_imm32,
                        int_aarch64_neon_vsri> {
    let Inst{22-21} = 0b01;
  }

  def _2D : N2VShiftIns<0b1, u, opcode, asmop, "2d", VPR128, v2i64, shr_imm64,
                        int_aarch64_neon_vsri> {
    let Inst{22} = 0b1;
  }
}

// Shift left and insert
defm SLIvvi   : NeonI_N2VShLIns<0b1, 0b01010, "sli">;

// Shift right and insert
defm SRIvvi   : NeonI_N2VShRIns<0b1, 0b01000, "sri">;

class N2VShR_Narrow<bit q, bit u, bits<5> opcode, string asmop, string DestT,
                    string SrcT, Operand ImmTy>
  : NeonI_2VShiftImm<q, u, opcode,
                     (outs VPR64:$Rd), (ins VPR128:$Rn, ImmTy:$Imm),
                     asmop # "\t$Rd." # DestT # ", $Rn." # SrcT # ", $Imm",
                     [], NoItinerary>;

class N2VShR_Narrow_Hi<bit q, bit u, bits<5> opcode, string asmop, string DestT,
                       string SrcT, Operand ImmTy>
  : NeonI_2VShiftImm<q, u, opcode, (outs VPR128:$Rd),
                     (ins VPR128:$src, VPR128:$Rn, ImmTy:$Imm),
                     asmop # "\t$Rd." # DestT # ", $Rn." # SrcT # ", $Imm",
                     [], NoItinerary> {
  let Constraints = "$src = $Rd";
}

// left long shift by immediate
multiclass NeonI_N2VShR_Narrow<bit u, bits<5> opcode, string asmop> {
  def _8B : N2VShR_Narrow<0b0, u, opcode, asmop, "8b", "8h", shr_imm8> {
    let Inst{22-19} = 0b0001;
  }

  def _4H : N2VShR_Narrow<0b0, u, opcode, asmop, "4h", "4s", shr_imm16> {
    let Inst{22-20} = 0b001;
  }

  def _2S : N2VShR_Narrow<0b0, u, opcode, asmop, "2s", "2d", shr_imm32> {
    let Inst{22-21} = 0b01;
  }

  // Shift Narrow High
  def _16B : N2VShR_Narrow_Hi<0b1, u, opcode, asmop # "2", "16b", "8h",
                              shr_imm8> {
    let Inst{22-19} = 0b0001;
  }

  def _8H : N2VShR_Narrow_Hi<0b1, u, opcode, asmop # "2", "8h", "4s",
                             shr_imm16> {
    let Inst{22-20} = 0b001;
  }

  def _4S : N2VShR_Narrow_Hi<0b1, u, opcode, asmop # "2", "4s", "2d",
                             shr_imm32> {
    let Inst{22-21} = 0b01;
  }
}

// Shift right narrow
defm SHRNvvi : NeonI_N2VShR_Narrow<0b0, 0b10000, "shrn">;

// Shift right narrow (prefix Q is saturating, prefix R is rounding)
defm QSHRUNvvi :NeonI_N2VShR_Narrow<0b1, 0b10000, "sqshrun">;
defm RSHRNvvi : NeonI_N2VShR_Narrow<0b0, 0b10001, "rshrn">;
defm QRSHRUNvvi : NeonI_N2VShR_Narrow<0b1, 0b10001, "sqrshrun">;
defm SQSHRNvvi : NeonI_N2VShR_Narrow<0b0, 0b10010, "sqshrn">;
defm UQSHRNvvi : NeonI_N2VShR_Narrow<0b1, 0b10010, "uqshrn">;
defm SQRSHRNvvi : NeonI_N2VShR_Narrow<0b0, 0b10011, "sqrshrn">;
defm UQRSHRNvvi : NeonI_N2VShR_Narrow<0b1, 0b10011, "uqrshrn">;

def Neon_combine_2D : PatFrag<(ops node:$Rm, node:$Rn),
                              (v2i64 (concat_vectors (v1i64 node:$Rm),
                                                     (v1i64 node:$Rn)))>;
def Neon_combine_8H : PatFrag<(ops node:$Rm, node:$Rn),
                              (v8i16 (concat_vectors (v4i16 node:$Rm),
                                                     (v4i16 node:$Rn)))>;
def Neon_combine_4S : PatFrag<(ops node:$Rm, node:$Rn),
                              (v4i32 (concat_vectors (v2i32 node:$Rm),
                                                     (v2i32 node:$Rn)))>;
def Neon_combine_4f : PatFrag<(ops node:$Rm, node:$Rn),
                              (v4f32 (concat_vectors (v2f32 node:$Rm),
                                                     (v2f32 node:$Rn)))>;
def Neon_combine_2d : PatFrag<(ops node:$Rm, node:$Rn),
                              (v2f64 (concat_vectors (v1f64 node:$Rm),
                                                     (v1f64 node:$Rn)))>;

def Neon_lshrImm8H : PatFrag<(ops node:$lhs, node:$rhs),
                             (v8i16 (srl (v8i16 node:$lhs),
                               (v8i16 (Neon_vdup (i32 node:$rhs)))))>;
def Neon_lshrImm4S : PatFrag<(ops node:$lhs, node:$rhs),
                             (v4i32 (srl (v4i32 node:$lhs),
                               (v4i32 (Neon_vdup (i32 node:$rhs)))))>;
def Neon_lshrImm2D : PatFrag<(ops node:$lhs, node:$rhs),
                             (v2i64 (srl (v2i64 node:$lhs),
                               (v2i64 (Neon_vdup (i32 node:$rhs)))))>;
def Neon_ashrImm8H : PatFrag<(ops node:$lhs, node:$rhs),
                             (v8i16 (sra (v8i16 node:$lhs),
                               (v8i16 (Neon_vdup (i32 node:$rhs)))))>;
def Neon_ashrImm4S : PatFrag<(ops node:$lhs, node:$rhs),
                             (v4i32 (sra (v4i32 node:$lhs),
                               (v4i32 (Neon_vdup (i32 node:$rhs)))))>;
def Neon_ashrImm2D : PatFrag<(ops node:$lhs, node:$rhs),
                             (v2i64 (sra (v2i64 node:$lhs),
                               (v2i64 (Neon_vdup (i32 node:$rhs)))))>;

// Normal shift right narrow is matched by IR (srl/sra, trunc, concat_vectors)
multiclass Neon_shiftNarrow_patterns<string shr> {
  def : Pat<(v8i8 (trunc (!cast<PatFrag>("Neon_" # shr # "Imm8H") VPR128:$Rn,
              (i32 imm:$Imm)))),
            (SHRNvvi_8B VPR128:$Rn, imm:$Imm)>;
  def : Pat<(v4i16 (trunc (!cast<PatFrag>("Neon_" # shr # "Imm4S") VPR128:$Rn,
              (i32 imm:$Imm)))),
            (SHRNvvi_4H VPR128:$Rn, imm:$Imm)>;
  def : Pat<(v2i32 (trunc (!cast<PatFrag>("Neon_" # shr # "Imm2D") VPR128:$Rn,
              (i32 imm:$Imm)))),
            (SHRNvvi_2S VPR128:$Rn, imm:$Imm)>;

  def : Pat<(Neon_combine_2D (v1i64 VPR64:$src), (v1i64 (bitconvert
              (v8i8 (trunc (!cast<PatFrag>("Neon_" # shr # "Imm8H")
                VPR128:$Rn, (i32 imm:$Imm))))))),
            (SHRNvvi_16B (v2i64 (SUBREG_TO_REG (i64 0), VPR64:$src, sub_64)),
                         VPR128:$Rn, imm:$Imm)>;
  def : Pat<(Neon_combine_2D (v1i64 VPR64:$src), (v1i64 (bitconvert
              (v4i16 (trunc (!cast<PatFrag>("Neon_" # shr # "Imm4S")
                VPR128:$Rn, (i32 imm:$Imm))))))),
            (SHRNvvi_8H (SUBREG_TO_REG (i64 0), VPR64:$src, sub_64),
                        VPR128:$Rn, imm:$Imm)>;
  def : Pat<(Neon_combine_2D (v1i64 VPR64:$src), (v1i64 (bitconvert
              (v2i32 (trunc (!cast<PatFrag>("Neon_" # shr # "Imm2D")
                VPR128:$Rn, (i32 imm:$Imm))))))),
            (SHRNvvi_4S (SUBREG_TO_REG (i64 0), VPR64:$src, sub_64),
                        VPR128:$Rn, imm:$Imm)>;
}

multiclass Neon_shiftNarrow_QR_patterns<SDPatternOperator op, string prefix> {
  def : Pat<(v8i8 (op (v8i16 VPR128:$Rn), imm:$Imm)),
            (!cast<Instruction>(prefix # "_8B") VPR128:$Rn, imm:$Imm)>;
  def : Pat<(v4i16 (op (v4i32 VPR128:$Rn), imm:$Imm)),
            (!cast<Instruction>(prefix # "_4H") VPR128:$Rn, imm:$Imm)>;
  def : Pat<(v2i32 (op (v2i64 VPR128:$Rn), imm:$Imm)),
            (!cast<Instruction>(prefix # "_2S") VPR128:$Rn, imm:$Imm)>;

  def : Pat<(Neon_combine_2D (v1i64 VPR64:$src),
                (v1i64 (bitconvert (v8i8 (op (v8i16 VPR128:$Rn), imm:$Imm))))),
            (!cast<Instruction>(prefix # "_16B")
                (SUBREG_TO_REG (i64 0), VPR64:$src, sub_64),
                VPR128:$Rn, imm:$Imm)>;
  def : Pat<(Neon_combine_2D (v1i64 VPR64:$src),
                (v1i64 (bitconvert (v4i16 (op (v4i32 VPR128:$Rn), imm:$Imm))))),
            (!cast<Instruction>(prefix # "_8H")
                (SUBREG_TO_REG (i64 0), VPR64:$src, sub_64),
                VPR128:$Rn, imm:$Imm)>;
  def : Pat<(Neon_combine_2D (v1i64 VPR64:$src),
                (v1i64 (bitconvert (v2i32 (op (v2i64 VPR128:$Rn), imm:$Imm))))),
            (!cast<Instruction>(prefix # "_4S")
                  (SUBREG_TO_REG (i64 0), VPR64:$src, sub_64),
                  VPR128:$Rn, imm:$Imm)>;
}

defm : Neon_shiftNarrow_patterns<"lshr">;
defm : Neon_shiftNarrow_patterns<"ashr">;

defm : Neon_shiftNarrow_QR_patterns<int_aarch64_neon_vsqshrun, "QSHRUNvvi">;
defm : Neon_shiftNarrow_QR_patterns<int_aarch64_neon_vrshrn, "RSHRNvvi">;
defm : Neon_shiftNarrow_QR_patterns<int_aarch64_neon_vsqrshrun, "QRSHRUNvvi">;
defm : Neon_shiftNarrow_QR_patterns<int_aarch64_neon_vsqshrn, "SQSHRNvvi">;
defm : Neon_shiftNarrow_QR_patterns<int_aarch64_neon_vuqshrn, "UQSHRNvvi">;
defm : Neon_shiftNarrow_QR_patterns<int_aarch64_neon_vsqrshrn, "SQRSHRNvvi">;
defm : Neon_shiftNarrow_QR_patterns<int_aarch64_neon_vuqrshrn, "UQRSHRNvvi">;

// Convert fix-point and float-pointing
class N2VCvt_Fx<bit q, bit u, bits<5> opcode, string asmop, string T,
                RegisterOperand VPRC, ValueType DestTy, ValueType SrcTy,
                Operand ImmTy, SDPatternOperator IntOp>
  : NeonI_2VShiftImm<q, u, opcode,
                     (outs VPRC:$Rd), (ins VPRC:$Rn, ImmTy:$Imm),
                     asmop # "\t$Rd." # T # ", $Rn." # T # ", $Imm",
                     [(set (DestTy VPRC:$Rd), (DestTy (IntOp (SrcTy VPRC:$Rn),
                       (i32 imm:$Imm))))],
                     NoItinerary>;

multiclass NeonI_N2VCvt_Fx2fp<bit u, bits<5> opcode, string asmop,
                              SDPatternOperator IntOp> {
  def _2S : N2VCvt_Fx<0, u, opcode, asmop, "2s", VPR64, v2f32, v2i32,
                      shr_imm32, IntOp> {
    let Inst{22-21} = 0b01;
  }

  def _4S : N2VCvt_Fx<1, u, opcode, asmop, "4s", VPR128, v4f32, v4i32,
                      shr_imm32, IntOp> {
    let Inst{22-21} = 0b01;
  }

  def _2D : N2VCvt_Fx<1, u, opcode, asmop, "2d", VPR128, v2f64, v2i64,
                      shr_imm64, IntOp> {
    let Inst{22} = 0b1;
  }
}

multiclass NeonI_N2VCvt_Fp2fx<bit u, bits<5> opcode, string asmop,
                              SDPatternOperator IntOp> {
  def _2S : N2VCvt_Fx<0, u, opcode, asmop, "2s", VPR64, v2i32, v2f32,
                      shr_imm32, IntOp> {
    let Inst{22-21} = 0b01;
  }

  def _4S : N2VCvt_Fx<1, u, opcode, asmop, "4s", VPR128, v4i32, v4f32,
                      shr_imm32, IntOp> {
    let Inst{22-21} = 0b01;
  }

  def _2D : N2VCvt_Fx<1, u, opcode, asmop, "2d", VPR128, v2i64, v2f64,
                      shr_imm64, IntOp> {
    let Inst{22} = 0b1;
  }
}

// Convert fixed-point to floating-point
defm VCVTxs2f : NeonI_N2VCvt_Fx2fp<0, 0b11100, "scvtf",
                                   int_arm_neon_vcvtfxs2fp>;
defm VCVTxu2f : NeonI_N2VCvt_Fx2fp<1, 0b11100, "ucvtf",
                                   int_arm_neon_vcvtfxu2fp>;

// Convert floating-point to fixed-point
defm VCVTf2xs : NeonI_N2VCvt_Fp2fx<0, 0b11111, "fcvtzs",
                                   int_arm_neon_vcvtfp2fxs>;
defm VCVTf2xu : NeonI_N2VCvt_Fp2fx<1, 0b11111, "fcvtzu",
                                   int_arm_neon_vcvtfp2fxu>;

multiclass Neon_sshll2_0<SDNode ext>
{
  def _v8i8  : PatFrag<(ops node:$Rn),
                       (v8i16 (ext (v8i8 (Neon_High16B node:$Rn))))>;
  def _v4i16 : PatFrag<(ops node:$Rn),
                       (v4i32 (ext (v4i16 (Neon_High8H node:$Rn))))>;
  def _v2i32 : PatFrag<(ops node:$Rn),
                       (v2i64 (ext (v2i32 (Neon_High4S node:$Rn))))>;
}

defm NI_sext_high : Neon_sshll2_0<sext>;
defm NI_zext_high : Neon_sshll2_0<zext>;


//===----------------------------------------------------------------------===//
// Multiclasses for NeonI_Across
//===----------------------------------------------------------------------===//

// Variant 1

multiclass NeonI_2VAcross_1<bit u, bits<5> opcode,
                            string asmop, SDPatternOperator opnode>
{
    def _1h8b:  NeonI_2VAcross<0b0, u, 0b00, opcode,
                (outs FPR16:$Rd), (ins VPR64:$Rn),
                asmop # "\t$Rd, $Rn.8b",
                [(set (v1i16 FPR16:$Rd),
                    (v1i16 (opnode (v8i8 VPR64:$Rn))))],
                NoItinerary>;

    def _1h16b: NeonI_2VAcross<0b1, u, 0b00, opcode,
                (outs FPR16:$Rd), (ins VPR128:$Rn),
                asmop # "\t$Rd, $Rn.16b",
                [(set (v1i16 FPR16:$Rd),
                    (v1i16 (opnode (v16i8 VPR128:$Rn))))],
                NoItinerary>;

    def _1s4h:  NeonI_2VAcross<0b0, u, 0b01, opcode,
                (outs FPR32:$Rd), (ins VPR64:$Rn),
                asmop # "\t$Rd, $Rn.4h",
                [(set (v1i32 FPR32:$Rd),
                    (v1i32 (opnode (v4i16 VPR64:$Rn))))],
                NoItinerary>;

    def _1s8h:  NeonI_2VAcross<0b1, u, 0b01, opcode,
                (outs FPR32:$Rd), (ins VPR128:$Rn),
                asmop # "\t$Rd, $Rn.8h",
                [(set (v1i32 FPR32:$Rd),
                    (v1i32 (opnode (v8i16 VPR128:$Rn))))],
                NoItinerary>;

    // _1d2s doesn't exist!

    def _1d4s:  NeonI_2VAcross<0b1, u, 0b10, opcode,
                (outs FPR64:$Rd), (ins VPR128:$Rn),
                asmop # "\t$Rd, $Rn.4s",
                [(set (v1i64 FPR64:$Rd),
                    (v1i64 (opnode (v4i32 VPR128:$Rn))))],
                NoItinerary>;
}

defm SADDLV : NeonI_2VAcross_1<0b0, 0b00011, "saddlv", int_aarch64_neon_saddlv>;
defm UADDLV : NeonI_2VAcross_1<0b1, 0b00011, "uaddlv", int_aarch64_neon_uaddlv>;

// Variant 2

multiclass NeonI_2VAcross_2<bit u, bits<5> opcode,
                            string asmop, SDPatternOperator opnode>
{
    def _1b8b:  NeonI_2VAcross<0b0, u, 0b00, opcode,
                (outs FPR8:$Rd), (ins VPR64:$Rn),
                asmop # "\t$Rd, $Rn.8b",
                [(set (v1i8 FPR8:$Rd),
                    (v1i8 (opnode (v8i8 VPR64:$Rn))))],
                NoItinerary>;

    def _1b16b: NeonI_2VAcross<0b1, u, 0b00, opcode,
                (outs FPR8:$Rd), (ins VPR128:$Rn),
                asmop # "\t$Rd, $Rn.16b",
                [(set (v1i8 FPR8:$Rd),
                    (v1i8 (opnode (v16i8 VPR128:$Rn))))],
                NoItinerary>;

    def _1h4h:  NeonI_2VAcross<0b0, u, 0b01, opcode,
                (outs FPR16:$Rd), (ins VPR64:$Rn),
                asmop # "\t$Rd, $Rn.4h",
                [(set (v1i16 FPR16:$Rd),
                    (v1i16 (opnode (v4i16 VPR64:$Rn))))],
                NoItinerary>;

    def _1h8h:  NeonI_2VAcross<0b1, u, 0b01, opcode,
                (outs FPR16:$Rd), (ins VPR128:$Rn),
                asmop # "\t$Rd, $Rn.8h",
                [(set (v1i16 FPR16:$Rd),
                    (v1i16 (opnode (v8i16 VPR128:$Rn))))],
                NoItinerary>;

    // _1s2s doesn't exist!

    def _1s4s:  NeonI_2VAcross<0b1, u, 0b10, opcode,
                (outs FPR32:$Rd), (ins VPR128:$Rn),
                asmop # "\t$Rd, $Rn.4s",
                [(set (v1i32 FPR32:$Rd),
                    (v1i32 (opnode (v4i32 VPR128:$Rn))))],
                NoItinerary>;
}

defm SMAXV : NeonI_2VAcross_2<0b0, 0b01010, "smaxv", int_aarch64_neon_smaxv>;
defm UMAXV : NeonI_2VAcross_2<0b1, 0b01010, "umaxv", int_aarch64_neon_umaxv>;

defm SMINV : NeonI_2VAcross_2<0b0, 0b11010, "sminv", int_aarch64_neon_sminv>;
defm UMINV : NeonI_2VAcross_2<0b1, 0b11010, "uminv", int_aarch64_neon_uminv>;

defm ADDV : NeonI_2VAcross_2<0b0, 0b11011, "addv", int_aarch64_neon_vaddv>;

// Variant 3

multiclass NeonI_2VAcross_3<bit u, bits<5> opcode, bits<2> size,
                            string asmop, SDPatternOperator opnode>
{
    def _1s4s:  NeonI_2VAcross<0b1, u, size, opcode,
                (outs FPR32:$Rd), (ins VPR128:$Rn),
                asmop # "\t$Rd, $Rn.4s",
                [(set (v1f32 FPR32:$Rd),
                    (v1f32 (opnode (v4f32 VPR128:$Rn))))],
                NoItinerary>;
}

defm FMAXNMV : NeonI_2VAcross_3<0b1, 0b01100, 0b00, "fmaxnmv",
                                int_aarch64_neon_vmaxnmv>;
defm FMINNMV : NeonI_2VAcross_3<0b1, 0b01100, 0b10, "fminnmv",
                                int_aarch64_neon_vminnmv>;

defm FMAXV : NeonI_2VAcross_3<0b1, 0b01111, 0b00, "fmaxv",
                              int_aarch64_neon_vmaxv>;
defm FMINV : NeonI_2VAcross_3<0b1, 0b01111, 0b10, "fminv",
                              int_aarch64_neon_vminv>;

// The followings are for instruction class (3V Diff)

// normal long/long2 pattern
class NeonI_3VDL<bit q, bit u, bits<2> size, bits<4> opcode,
                 string asmop, string ResS, string OpS,
                 SDPatternOperator opnode, SDPatternOperator ext,
                 RegisterOperand OpVPR,
                 ValueType ResTy, ValueType OpTy>
  : NeonI_3VDiff<q, u, size, opcode,
                 (outs VPR128:$Rd), (ins OpVPR:$Rn, OpVPR:$Rm),
                 asmop # "\t$Rd." # ResS # ", $Rn." # OpS # ", $Rm." # OpS,
                 [(set (ResTy VPR128:$Rd),
                    (ResTy (opnode (ResTy (ext (OpTy OpVPR:$Rn))),
                                   (ResTy (ext (OpTy OpVPR:$Rm))))))],
                 NoItinerary>;

multiclass NeonI_3VDL_s<bit u, bits<4> opcode,
                        string asmop, SDPatternOperator opnode,
                        bit Commutable = 0>
{
  let isCommutable = Commutable in {
    def _8h8b : NeonI_3VDL<0b0, u, 0b00, opcode, asmop, "8h", "8b",
                           opnode, sext, VPR64, v8i16, v8i8>;
    def _4s4h : NeonI_3VDL<0b0, u, 0b01, opcode, asmop, "4s", "4h",
                           opnode, sext, VPR64, v4i32, v4i16>;
    def _2d2s : NeonI_3VDL<0b0, u, 0b10, opcode, asmop, "2d", "2s",
                           opnode, sext, VPR64, v2i64, v2i32>;
  }
}

multiclass NeonI_3VDL2_s<bit u, bits<4> opcode,
                         string asmop, SDPatternOperator opnode,
                         bit Commutable = 0>
{
  let isCommutable = Commutable in {
    def _8h16b : NeonI_3VDL<0b1, u, 0b00, opcode, asmop, "8h", "16b",
                            opnode, NI_sext_high_v8i8, VPR128, v8i16, v16i8>;
    def _4s8h  : NeonI_3VDL<0b1, u, 0b01, opcode, asmop, "4s", "8h",
                            opnode, NI_sext_high_v4i16, VPR128, v4i32, v8i16>;
    def _2d4s  : NeonI_3VDL<0b1, u, 0b10, opcode, asmop, "2d", "4s",
                            opnode, NI_sext_high_v2i32, VPR128, v2i64, v4i32>;
  }
}

multiclass NeonI_3VDL_u<bit u, bits<4> opcode,
                          string asmop, SDPatternOperator opnode,
                          bit Commutable = 0>
{
  let isCommutable = Commutable in {
    def _8h8b : NeonI_3VDL<0b0, u, 0b00, opcode, asmop, "8h", "8b",
                           opnode, zext, VPR64, v8i16, v8i8>;
    def _4s4h : NeonI_3VDL<0b0, u, 0b01, opcode, asmop, "4s", "4h",
                           opnode, zext, VPR64, v4i32, v4i16>;
    def _2d2s : NeonI_3VDL<0b0, u, 0b10, opcode, asmop, "2d", "2s",
                           opnode, zext, VPR64, v2i64, v2i32>;
  }
}

multiclass NeonI_3VDL2_u<bit u, bits<4> opcode,
                           string asmop, SDPatternOperator opnode,
                           bit Commutable = 0>
{
  let isCommutable = Commutable in {
    def _8h16b : NeonI_3VDL<0b1, u, 0b00, opcode, asmop, "8h", "16b",
                            opnode, NI_zext_high_v8i8, VPR128, v8i16, v16i8>;
    def _4s8h : NeonI_3VDL<0b1, u, 0b01, opcode, asmop, "4s", "8h",
                           opnode, NI_zext_high_v4i16, VPR128, v4i32, v8i16>;
    def _2d4s : NeonI_3VDL<0b1, u, 0b10, opcode, asmop, "2d", "4s",
                           opnode, NI_zext_high_v2i32, VPR128, v2i64, v4i32>;
  }
}

defm SADDLvvv :  NeonI_3VDL_s<0b0, 0b0000, "saddl", add, 1>;
defm UADDLvvv :  NeonI_3VDL_u<0b1, 0b0000, "uaddl", add, 1>;

defm SADDL2vvv :  NeonI_3VDL2_s<0b0, 0b0000, "saddl2", add, 1>;
defm UADDL2vvv :  NeonI_3VDL2_u<0b1, 0b0000, "uaddl2", add, 1>;

defm SSUBLvvv :  NeonI_3VDL_s<0b0, 0b0010, "ssubl", sub, 0>;
defm USUBLvvv :  NeonI_3VDL_u<0b1, 0b0010, "usubl", sub, 0>;

defm SSUBL2vvv :  NeonI_3VDL2_s<0b0, 0b0010, "ssubl2", sub, 0>;
defm USUBL2vvv :  NeonI_3VDL2_u<0b1, 0b0010, "usubl2", sub, 0>;

// normal wide/wide2 pattern
class NeonI_3VDW<bit q, bit u, bits<2> size, bits<4> opcode,
                 string asmop, string ResS, string OpS,
                 SDPatternOperator opnode, SDPatternOperator ext,
                 RegisterOperand OpVPR,
                 ValueType ResTy, ValueType OpTy>
  : NeonI_3VDiff<q, u, size, opcode,
                 (outs VPR128:$Rd), (ins VPR128:$Rn, OpVPR:$Rm),
                 asmop # "\t$Rd." # ResS # ", $Rn." # ResS # ", $Rm." # OpS,
                 [(set (ResTy VPR128:$Rd),
                    (ResTy (opnode (ResTy VPR128:$Rn),
                                   (ResTy (ext (OpTy OpVPR:$Rm))))))],
                 NoItinerary>;

multiclass NeonI_3VDW_s<bit u, bits<4> opcode,
                        string asmop, SDPatternOperator opnode>
{
  def _8h8b : NeonI_3VDW<0b0, u, 0b00, opcode, asmop, "8h", "8b",
                         opnode, sext, VPR64, v8i16, v8i8>;
  def _4s4h : NeonI_3VDW<0b0, u, 0b01, opcode, asmop, "4s", "4h",
                         opnode, sext, VPR64, v4i32, v4i16>;
  def _2d2s : NeonI_3VDW<0b0, u, 0b10, opcode, asmop, "2d", "2s",
                         opnode, sext, VPR64, v2i64, v2i32>;
}

defm SADDWvvv :  NeonI_3VDW_s<0b0, 0b0001, "saddw", add>;
defm SSUBWvvv :  NeonI_3VDW_s<0b0, 0b0011, "ssubw", sub>;

multiclass NeonI_3VDW2_s<bit u, bits<4> opcode,
                         string asmop, SDPatternOperator opnode>
{
  def _8h16b : NeonI_3VDW<0b1, u, 0b00, opcode, asmop, "8h", "16b",
                          opnode, NI_sext_high_v8i8, VPR128, v8i16, v16i8>;
  def _4s8h  : NeonI_3VDW<0b1, u, 0b01, opcode, asmop, "4s", "8h",
                          opnode, NI_sext_high_v4i16, VPR128, v4i32, v8i16>;
  def _2d4s  : NeonI_3VDW<0b1, u, 0b10, opcode, asmop, "2d", "4s",
                          opnode, NI_sext_high_v2i32, VPR128, v2i64, v4i32>;
}

defm SADDW2vvv :  NeonI_3VDW2_s<0b0, 0b0001, "saddw2", add>;
defm SSUBW2vvv :  NeonI_3VDW2_s<0b0, 0b0011, "ssubw2", sub>;

multiclass NeonI_3VDW_u<bit u, bits<4> opcode,
                        string asmop, SDPatternOperator opnode>
{
  def _8h8b : NeonI_3VDW<0b0, u, 0b00, opcode, asmop, "8h", "8b",
                         opnode, zext, VPR64, v8i16, v8i8>;
  def _4s4h : NeonI_3VDW<0b0, u, 0b01, opcode, asmop, "4s", "4h",
                         opnode, zext, VPR64, v4i32, v4i16>;
  def _2d2s : NeonI_3VDW<0b0, u, 0b10, opcode, asmop, "2d", "2s",
                         opnode, zext, VPR64, v2i64, v2i32>;
}

defm UADDWvvv :  NeonI_3VDW_u<0b1, 0b0001, "uaddw", add>;
defm USUBWvvv :  NeonI_3VDW_u<0b1, 0b0011, "usubw", sub>;

multiclass NeonI_3VDW2_u<bit u, bits<4> opcode,
                           string asmop, SDPatternOperator opnode>
{
  def _8h16b : NeonI_3VDW<0b1, u, 0b00, opcode, asmop, "8h", "16b",
                          opnode, NI_zext_high_v8i8, VPR128, v8i16, v16i8>;
  def _4s8h : NeonI_3VDW<0b1, u, 0b01, opcode, asmop, "4s", "8h",
                         opnode, NI_zext_high_v4i16, VPR128, v4i32, v8i16>;
  def _2d4s : NeonI_3VDW<0b1, u, 0b10, opcode, asmop, "2d", "4s",
                         opnode, NI_zext_high_v2i32, VPR128, v2i64, v4i32>;
}

defm UADDW2vvv :  NeonI_3VDW2_u<0b1, 0b0001, "uaddw2", add>;
defm USUBW2vvv :  NeonI_3VDW2_u<0b1, 0b0011, "usubw2", sub>;

// Get the high half part of the vector element.
multiclass NeonI_get_high
{
  def _8h : PatFrag<(ops node:$Rn),
                    (v8i8 (trunc (v8i16 (srl (v8i16 node:$Rn),
                                             (v8i16 (Neon_vdup (i32 8)))))))>;
  def _4s : PatFrag<(ops node:$Rn),
                    (v4i16 (trunc (v4i32 (srl (v4i32 node:$Rn),
                                              (v4i32 (Neon_vdup (i32 16)))))))>;
  def _2d : PatFrag<(ops node:$Rn),
                    (v2i32 (trunc (v2i64 (srl (v2i64 node:$Rn),
                                              (v2i64 (Neon_vdup (i32 32)))))))>;
}

defm NI_get_hi : NeonI_get_high;

// pattern for addhn/subhn with 2 operands
class NeonI_3VDN_addhn_2Op<bit q, bit u, bits<2> size, bits<4> opcode,
                           string asmop, string ResS, string OpS,
                           SDPatternOperator opnode, SDPatternOperator get_hi,
                           ValueType ResTy, ValueType OpTy>
  : NeonI_3VDiff<q, u, size, opcode,
                 (outs VPR64:$Rd), (ins VPR128:$Rn, VPR128:$Rm),
                 asmop # "\t$Rd." # ResS # ", $Rn." # OpS # ", $Rm." # OpS,
                 [(set (ResTy VPR64:$Rd),
                    (ResTy (get_hi
                      (OpTy (opnode (OpTy VPR128:$Rn),
                                    (OpTy VPR128:$Rm))))))],
                 NoItinerary>;

multiclass NeonI_3VDN_addhn_2Op<bit u, bits<4> opcode,
                                string asmop, SDPatternOperator opnode,
                                bit Commutable = 0>
{
  let isCommutable = Commutable in {
    def _8b8h : NeonI_3VDN_addhn_2Op<0b0, u, 0b00, opcode, asmop, "8b", "8h",
                                     opnode, NI_get_hi_8h, v8i8, v8i16>;
    def _4h4s : NeonI_3VDN_addhn_2Op<0b0, u, 0b01, opcode, asmop, "4h", "4s",
                                     opnode, NI_get_hi_4s, v4i16, v4i32>;
    def _2s2d : NeonI_3VDN_addhn_2Op<0b0, u, 0b10, opcode, asmop, "2s", "2d",
                                     opnode, NI_get_hi_2d, v2i32, v2i64>;
  }
}

defm ADDHNvvv  : NeonI_3VDN_addhn_2Op<0b0, 0b0100, "addhn", add, 1>;
defm SUBHNvvv  : NeonI_3VDN_addhn_2Op<0b0, 0b0110, "subhn", sub, 0>;

// pattern for operation with 2 operands
class NeonI_3VD_2Op<bit q, bit u, bits<2> size, bits<4> opcode,
                    string asmop, string ResS, string OpS,
                    SDPatternOperator opnode,
                    RegisterOperand ResVPR, RegisterOperand OpVPR,
                    ValueType ResTy, ValueType OpTy>
  : NeonI_3VDiff<q, u, size, opcode,
                 (outs ResVPR:$Rd), (ins OpVPR:$Rn, OpVPR:$Rm),
                 asmop # "\t$Rd." # ResS # ", $Rn." # OpS # ", $Rm." # OpS,
                 [(set (ResTy ResVPR:$Rd),
                    (ResTy (opnode (OpTy OpVPR:$Rn), (OpTy OpVPR:$Rm))))],
                 NoItinerary>;

// normal narrow pattern
multiclass NeonI_3VDN_2Op<bit u, bits<4> opcode,
                          string asmop, SDPatternOperator opnode,
                          bit Commutable = 0>
{
  let isCommutable = Commutable in {
    def _8b8h : NeonI_3VD_2Op<0b0, u, 0b00, opcode, asmop, "8b", "8h",
                              opnode, VPR64, VPR128, v8i8, v8i16>;
    def _4h4s : NeonI_3VD_2Op<0b0, u, 0b01, opcode, asmop, "4h", "4s",
                              opnode, VPR64, VPR128, v4i16, v4i32>;
    def _2s2d : NeonI_3VD_2Op<0b0, u, 0b10, opcode, asmop, "2s", "2d",
                              opnode, VPR64, VPR128, v2i32, v2i64>;
  }
}

defm RADDHNvvv : NeonI_3VDN_2Op<0b1, 0b0100, "raddhn", int_arm_neon_vraddhn, 1>;
defm RSUBHNvvv : NeonI_3VDN_2Op<0b1, 0b0110, "rsubhn", int_arm_neon_vrsubhn, 0>;

// pattern for acle intrinsic with 3 operands
class NeonI_3VDN_3Op<bit q, bit u, bits<2> size, bits<4> opcode,
                     string asmop, string ResS, string OpS>
  : NeonI_3VDiff<q, u, size, opcode,
                 (outs VPR128:$Rd), (ins VPR128:$src, VPR128:$Rn, VPR128:$Rm),
                 asmop # "\t$Rd." # ResS # ", $Rn." # OpS # ", $Rm." # OpS,
                 [], NoItinerary> {
  let Constraints = "$src = $Rd";
  let neverHasSideEffects = 1;
}

multiclass NeonI_3VDN_3Op_v1<bit u, bits<4> opcode,
                             string asmop> {
  def _16b8h : NeonI_3VDN_3Op<0b1, u, 0b00, opcode, asmop, "16b", "8h">;
  def _8h4s : NeonI_3VDN_3Op<0b1, u, 0b01, opcode, asmop, "8h", "4s">;
  def _4s2d : NeonI_3VDN_3Op<0b1, u, 0b10, opcode, asmop, "4s", "2d">;
}

defm ADDHN2vvv  : NeonI_3VDN_3Op_v1<0b0, 0b0100, "addhn2">;
defm SUBHN2vvv  : NeonI_3VDN_3Op_v1<0b0, 0b0110, "subhn2">;

defm RADDHN2vvv : NeonI_3VDN_3Op_v1<0b1, 0b0100, "raddhn2">;
defm RSUBHN2vvv : NeonI_3VDN_3Op_v1<0b1, 0b0110, "rsubhn2">;

// Patterns have to be separate because there's a SUBREG_TO_REG in the output
// part.
class NarrowHighHalfPat<Instruction INST, ValueType DstTy, ValueType SrcTy,
                        SDPatternOperator coreop>
  : Pat<(Neon_combine_2D (v1i64 VPR64:$src),
                      (v1i64 (bitconvert (DstTy (coreop (SrcTy VPR128:$Rn),
                                                        (SrcTy VPR128:$Rm)))))),
        (INST (SUBREG_TO_REG (i64 0), VPR64:$src, sub_64),
              VPR128:$Rn, VPR128:$Rm)>;

// addhn2 patterns
def : NarrowHighHalfPat<ADDHN2vvv_16b8h, v8i8,  v8i16,
          BinOpFrag<(NI_get_hi_8h (add node:$LHS, node:$RHS))>>;
def : NarrowHighHalfPat<ADDHN2vvv_8h4s,  v4i16, v4i32,
          BinOpFrag<(NI_get_hi_4s (add node:$LHS, node:$RHS))>>;
def : NarrowHighHalfPat<ADDHN2vvv_4s2d,  v2i32, v2i64,
          BinOpFrag<(NI_get_hi_2d (add node:$LHS, node:$RHS))>>;

// subhn2 patterns
def : NarrowHighHalfPat<SUBHN2vvv_16b8h, v8i8,  v8i16,
          BinOpFrag<(NI_get_hi_8h (sub node:$LHS, node:$RHS))>>;
def : NarrowHighHalfPat<SUBHN2vvv_8h4s,  v4i16, v4i32,
          BinOpFrag<(NI_get_hi_4s (sub node:$LHS, node:$RHS))>>;
def : NarrowHighHalfPat<SUBHN2vvv_4s2d,  v2i32, v2i64,
          BinOpFrag<(NI_get_hi_2d (sub node:$LHS, node:$RHS))>>;

// raddhn2 patterns
def : NarrowHighHalfPat<RADDHN2vvv_16b8h, v8i8,  v8i16, int_arm_neon_vraddhn>;
def : NarrowHighHalfPat<RADDHN2vvv_8h4s,  v4i16, v4i32, int_arm_neon_vraddhn>;
def : NarrowHighHalfPat<RADDHN2vvv_4s2d,  v2i32, v2i64, int_arm_neon_vraddhn>;

// rsubhn2 patterns
def : NarrowHighHalfPat<RSUBHN2vvv_16b8h, v8i8,  v8i16, int_arm_neon_vrsubhn>;
def : NarrowHighHalfPat<RSUBHN2vvv_8h4s,  v4i16, v4i32, int_arm_neon_vrsubhn>;
def : NarrowHighHalfPat<RSUBHN2vvv_4s2d,  v2i32, v2i64, int_arm_neon_vrsubhn>;

// pattern that need to extend result
class NeonI_3VDL_Ext<bit q, bit u, bits<2> size, bits<4> opcode,
                     string asmop, string ResS, string OpS,
                     SDPatternOperator opnode,
                     RegisterOperand OpVPR,
                     ValueType ResTy, ValueType OpTy, ValueType OpSTy>
  : NeonI_3VDiff<q, u, size, opcode,
                 (outs VPR128:$Rd), (ins OpVPR:$Rn, OpVPR:$Rm),
                 asmop # "\t$Rd." # ResS # ", $Rn." # OpS # ", $Rm." # OpS,
                 [(set (ResTy VPR128:$Rd),
                    (ResTy (zext (OpSTy (opnode (OpTy OpVPR:$Rn),
                                                (OpTy OpVPR:$Rm))))))],
                 NoItinerary>;

multiclass NeonI_3VDL_zext<bit u, bits<4> opcode,
                           string asmop, SDPatternOperator opnode,
                           bit Commutable = 0>
{
  let isCommutable = Commutable in {
    def _8h8b : NeonI_3VDL_Ext<0b0, u, 0b00, opcode, asmop, "8h", "8b",
                               opnode, VPR64, v8i16, v8i8, v8i8>;
    def _4s4h : NeonI_3VDL_Ext<0b0, u, 0b01, opcode, asmop, "4s", "4h",
                               opnode, VPR64, v4i32, v4i16, v4i16>;
    def _2d2s : NeonI_3VDL_Ext<0b0, u, 0b10, opcode, asmop, "2d", "2s",
                               opnode, VPR64, v2i64, v2i32, v2i32>;
  }
}

defm SABDLvvv : NeonI_3VDL_zext<0b0, 0b0111, "sabdl", int_arm_neon_vabds, 1>;
defm UABDLvvv : NeonI_3VDL_zext<0b1, 0b0111, "uabdl", int_arm_neon_vabdu, 1>;

multiclass NeonI_Op_High<SDPatternOperator op>
{
  def _16B : PatFrag<(ops node:$Rn, node:$Rm),
                     (op (v8i8 (Neon_High16B node:$Rn)), (v8i8 (Neon_High16B node:$Rm)))>;
  def _8H  : PatFrag<(ops node:$Rn, node:$Rm),
                     (op (v4i16 (Neon_High8H node:$Rn)), (v4i16 (Neon_High8H node:$Rm)))>;
  def _4S  : PatFrag<(ops node:$Rn, node:$Rm),
                     (op (v2i32 (Neon_High4S node:$Rn)), (v2i32 (Neon_High4S node:$Rm)))>;

}

defm NI_sabdl_hi : NeonI_Op_High<int_arm_neon_vabds>;
defm NI_uabdl_hi : NeonI_Op_High<int_arm_neon_vabdu>;
defm NI_smull_hi : NeonI_Op_High<int_arm_neon_vmulls>;
defm NI_umull_hi : NeonI_Op_High<int_arm_neon_vmullu>;
defm NI_qdmull_hi : NeonI_Op_High<int_arm_neon_vqdmull>;
defm NI_pmull_hi : NeonI_Op_High<int_arm_neon_vmullp>;

multiclass NeonI_3VDL_Abd_u<bit u, bits<4> opcode,
                            string asmop, string opnode,
                            bit Commutable = 0>
{
  let isCommutable = Commutable in {
    def _8h8b  : NeonI_3VDL_Ext<0b1, u, 0b00, opcode, asmop, "8h", "16b",
                                !cast<PatFrag>(opnode # "_16B"),
                                VPR128, v8i16, v16i8, v8i8>;
    def _4s4h  : NeonI_3VDL_Ext<0b1, u, 0b01, opcode, asmop, "4s", "8h",
                                !cast<PatFrag>(opnode # "_8H"),
                                VPR128, v4i32, v8i16, v4i16>;
    def _2d2s  : NeonI_3VDL_Ext<0b1, u, 0b10, opcode, asmop, "2d", "4s",
                                !cast<PatFrag>(opnode # "_4S"),
                                VPR128, v2i64, v4i32, v2i32>;
  }
}

defm SABDL2vvv : NeonI_3VDL_Abd_u<0b0, 0b0111, "sabdl2", "NI_sabdl_hi", 1>;
defm UABDL2vvv : NeonI_3VDL_Abd_u<0b1, 0b0111, "uabdl2", "NI_uabdl_hi", 1>;

// For pattern that need two operators being chained.
class NeonI_3VDL_Aba<bit q, bit u, bits<2> size, bits<4> opcode,
                     string asmop, string ResS, string OpS, 
                     SDPatternOperator opnode, SDPatternOperator subop,
                     RegisterOperand OpVPR,
                     ValueType ResTy, ValueType OpTy, ValueType OpSTy>
  : NeonI_3VDiff<q, u, size, opcode,
                 (outs VPR128:$Rd), (ins VPR128:$src, OpVPR:$Rn, OpVPR:$Rm),
                 asmop # "\t$Rd." # ResS # ", $Rn." # OpS # ", $Rm." # OpS, 
                 [(set (ResTy VPR128:$Rd),
                    (ResTy (opnode
                      (ResTy VPR128:$src), 
                      (ResTy (zext (OpSTy (subop (OpTy OpVPR:$Rn),
                                                 (OpTy OpVPR:$Rm))))))))],
                 NoItinerary> {
  let Constraints = "$src = $Rd";
}

multiclass NeonI_3VDL_Aba_v1<bit u, bits<4> opcode,
                             string asmop, SDPatternOperator opnode,
                             SDPatternOperator subop>
{
  def _8h8b : NeonI_3VDL_Aba<0b0, u, 0b00, opcode, asmop, "8h", "8b",
                             opnode, subop, VPR64, v8i16, v8i8, v8i8>;
  def _4s4h : NeonI_3VDL_Aba<0b0, u, 0b01, opcode, asmop, "4s", "4h",
                             opnode, subop, VPR64, v4i32, v4i16, v4i16>;
  def _2d2s : NeonI_3VDL_Aba<0b0, u, 0b10, opcode, asmop, "2d", "2s",
                             opnode, subop, VPR64, v2i64, v2i32, v2i32>;
}

defm SABALvvv :  NeonI_3VDL_Aba_v1<0b0, 0b0101, "sabal",
                                   add, int_arm_neon_vabds>;
defm UABALvvv :  NeonI_3VDL_Aba_v1<0b1, 0b0101, "uabal",
                                   add, int_arm_neon_vabdu>;

multiclass NeonI_3VDL2_Aba_v1<bit u, bits<4> opcode,
                              string asmop, SDPatternOperator opnode,
                              string subop>
{
  def _8h8b : NeonI_3VDL_Aba<0b1, u, 0b00, opcode, asmop, "8h", "16b",
                             opnode, !cast<PatFrag>(subop # "_16B"), 
                             VPR128, v8i16, v16i8, v8i8>;
  def _4s4h : NeonI_3VDL_Aba<0b1, u, 0b01, opcode, asmop, "4s", "8h",
                             opnode, !cast<PatFrag>(subop # "_8H"), 
                             VPR128, v4i32, v8i16, v4i16>;
  def _2d2s : NeonI_3VDL_Aba<0b1, u, 0b10, opcode, asmop, "2d", "4s",
                             opnode, !cast<PatFrag>(subop # "_4S"), 
                             VPR128, v2i64, v4i32, v2i32>;
}

defm SABAL2vvv :  NeonI_3VDL2_Aba_v1<0b0, 0b0101, "sabal2", add,
                                     "NI_sabdl_hi">;
defm UABAL2vvv :  NeonI_3VDL2_Aba_v1<0b1, 0b0101, "uabal2", add,
                                     "NI_uabdl_hi">;

// Long pattern with 2 operands
multiclass NeonI_3VDL_2Op<bit u, bits<4> opcode,
                          string asmop, SDPatternOperator opnode,
                          bit Commutable = 0>
{
  let isCommutable = Commutable in {
    def _8h8b : NeonI_3VD_2Op<0b0, u, 0b00, opcode, asmop, "8h", "8b",
                              opnode, VPR128, VPR64, v8i16, v8i8>;
    def _4s4h : NeonI_3VD_2Op<0b0, u, 0b01, opcode, asmop, "4s", "4h",
                              opnode, VPR128, VPR64, v4i32, v4i16>;
    def _2d2s : NeonI_3VD_2Op<0b0, u, 0b10, opcode, asmop, "2d", "2s",
                              opnode, VPR128, VPR64, v2i64, v2i32>;
  }
}

defm SMULLvvv :  NeonI_3VDL_2Op<0b0, 0b1100, "smull", int_arm_neon_vmulls, 1>;
defm UMULLvvv :  NeonI_3VDL_2Op<0b1, 0b1100, "umull", int_arm_neon_vmullu, 1>;

class NeonI_3VDL2_2Op_mull<bit q, bit u, bits<2> size, bits<4> opcode,
                           string asmop, string ResS, string OpS,
                           SDPatternOperator opnode,
                           ValueType ResTy, ValueType OpTy>
  : NeonI_3VDiff<q, u, size, opcode,
                 (outs VPR128:$Rd), (ins VPR128:$Rn, VPR128:$Rm),
                 asmop # "\t$Rd." # ResS # ", $Rn." # OpS # ", $Rm." # OpS,
                 [(set (ResTy VPR128:$Rd),
                    (ResTy (opnode (OpTy VPR128:$Rn), (OpTy VPR128:$Rm))))],
                 NoItinerary>;


multiclass NeonI_3VDL2_2Op_mull_v1<bit u, bits<4> opcode,
                                   string asmop, 
                                   string opnode,
                                   bit Commutable = 0>
{
  let isCommutable = Commutable in {
    def _8h16b : NeonI_3VDL2_2Op_mull<0b1, u, 0b00, opcode, asmop, "8h", "16b",
                                      !cast<PatFrag>(opnode # "_16B"),
                                      v8i16, v16i8>;
    def _4s8h : NeonI_3VDL2_2Op_mull<0b1, u, 0b01, opcode, asmop, "4s", "8h",
                                     !cast<PatFrag>(opnode # "_8H"),
                                     v4i32, v8i16>;
    def _2d4s : NeonI_3VDL2_2Op_mull<0b1, u, 0b10, opcode, asmop, "2d", "4s",
                                     !cast<PatFrag>(opnode # "_4S"),
                                     v2i64, v4i32>;
  }
}

defm SMULL2vvv : NeonI_3VDL2_2Op_mull_v1<0b0, 0b1100, "smull2",
                                         "NI_smull_hi", 1>;
defm UMULL2vvv : NeonI_3VDL2_2Op_mull_v1<0b1, 0b1100, "umull2",
                                         "NI_umull_hi", 1>;

// Long pattern with 3 operands
class NeonI_3VDL_3Op<bit q, bit u, bits<2> size, bits<4> opcode,
                     string asmop, string ResS, string OpS,
                     SDPatternOperator opnode,
                     ValueType ResTy, ValueType OpTy>
  : NeonI_3VDiff<q, u, size, opcode,
                 (outs VPR128:$Rd), (ins VPR128:$src, VPR64:$Rn, VPR64:$Rm),
                 asmop # "\t$Rd." # ResS # ", $Rn." # OpS # ", $Rm." # OpS,
                 [(set (ResTy VPR128:$Rd),
                    (ResTy (opnode
                      (ResTy VPR128:$src),
                      (OpTy VPR64:$Rn), (OpTy VPR64:$Rm))))],
               NoItinerary> {
  let Constraints = "$src = $Rd";
}

multiclass NeonI_3VDL_3Op_v1<bit u, bits<4> opcode,
                             string asmop, SDPatternOperator opnode>
{
  def _8h8b : NeonI_3VDL_3Op<0b0, u, 0b00, opcode, asmop, "8h", "8b",
                             opnode, v8i16, v8i8>;
  def _4s4h : NeonI_3VDL_3Op<0b0, u, 0b01, opcode, asmop, "4s", "4h",
                             opnode, v4i32, v4i16>;
  def _2d2s : NeonI_3VDL_3Op<0b0, u, 0b10, opcode, asmop, "2d", "2s",
                             opnode, v2i64, v2i32>;
}

def Neon_smlal : PatFrag<(ops node:$Rd, node:$Rn, node:$Rm),
                         (add node:$Rd,
                            (int_arm_neon_vmulls node:$Rn, node:$Rm))>;

def Neon_umlal : PatFrag<(ops node:$Rd, node:$Rn, node:$Rm),
                         (add node:$Rd,
                            (int_arm_neon_vmullu node:$Rn, node:$Rm))>;

def Neon_smlsl : PatFrag<(ops node:$Rd, node:$Rn, node:$Rm),
                         (sub node:$Rd,
                            (int_arm_neon_vmulls node:$Rn, node:$Rm))>;

def Neon_umlsl : PatFrag<(ops node:$Rd, node:$Rn, node:$Rm),
                         (sub node:$Rd,
                            (int_arm_neon_vmullu node:$Rn, node:$Rm))>;

defm SMLALvvv :  NeonI_3VDL_3Op_v1<0b0, 0b1000, "smlal", Neon_smlal>;
defm UMLALvvv :  NeonI_3VDL_3Op_v1<0b1, 0b1000, "umlal", Neon_umlal>;

defm SMLSLvvv :  NeonI_3VDL_3Op_v1<0b0, 0b1010, "smlsl", Neon_smlsl>;
defm UMLSLvvv :  NeonI_3VDL_3Op_v1<0b1, 0b1010, "umlsl", Neon_umlsl>;

class NeonI_3VDL2_3Op_mlas<bit q, bit u, bits<2> size, bits<4> opcode,
                           string asmop, string ResS, string OpS,
                           SDPatternOperator subop, SDPatternOperator opnode,
                           RegisterOperand OpVPR,
                           ValueType ResTy, ValueType OpTy>
  : NeonI_3VDiff<q, u, size, opcode,
               (outs VPR128:$Rd), (ins VPR128:$src, OpVPR:$Rn, OpVPR:$Rm),
               asmop # "\t$Rd." # ResS # ", $Rn." # OpS # ", $Rm." # OpS,
               [(set (ResTy VPR128:$Rd),
                  (ResTy (subop
                    (ResTy VPR128:$src),
                    (ResTy (opnode (OpTy OpVPR:$Rn), (OpTy OpVPR:$Rm))))))],
               NoItinerary> {
  let Constraints = "$src = $Rd";
}

multiclass NeonI_3VDL2_3Op_mlas_v1<bit u, bits<4> opcode,
                                   string asmop, 
                                   SDPatternOperator subop,
                                   string opnode>
{
  def _8h16b : NeonI_3VDL2_3Op_mlas<0b1, u, 0b00, opcode, asmop, "8h", "16b",
                                    subop, !cast<PatFrag>(opnode # "_16B"),
                                    VPR128, v8i16, v16i8>;
  def _4s8h : NeonI_3VDL2_3Op_mlas<0b1, u, 0b01, opcode, asmop, "4s", "8h",
                                   subop, !cast<PatFrag>(opnode # "_8H"), 
                                   VPR128, v4i32, v8i16>;
  def _2d4s : NeonI_3VDL2_3Op_mlas<0b1, u, 0b10, opcode, asmop, "2d", "4s",
                                   subop, !cast<PatFrag>(opnode # "_4S"),
                                   VPR128, v2i64, v4i32>;
}

defm SMLAL2vvv :  NeonI_3VDL2_3Op_mlas_v1<0b0, 0b1000, "smlal2",
                                          add, "NI_smull_hi">;
defm UMLAL2vvv :  NeonI_3VDL2_3Op_mlas_v1<0b1, 0b1000, "umlal2",
                                          add, "NI_umull_hi">;

defm SMLSL2vvv :  NeonI_3VDL2_3Op_mlas_v1<0b0, 0b1010, "smlsl2",
                                          sub, "NI_smull_hi">;
defm UMLSL2vvv :  NeonI_3VDL2_3Op_mlas_v1<0b1, 0b1010, "umlsl2",
                                          sub, "NI_umull_hi">;

multiclass NeonI_3VDL_qdmlal_3Op_v2<bit u, bits<4> opcode,
                                    string asmop, SDPatternOperator opnode>
{
  def _4s4h : NeonI_3VDL2_3Op_mlas<0b0, u, 0b01, opcode, asmop, "4s", "4h",
                                   opnode, int_arm_neon_vqdmull,
                                   VPR64, v4i32, v4i16>;
  def _2d2s : NeonI_3VDL2_3Op_mlas<0b0, u, 0b10, opcode, asmop, "2d", "2s",
                                   opnode, int_arm_neon_vqdmull,
                                   VPR64, v2i64, v2i32>;
}

defm SQDMLALvvv : NeonI_3VDL_qdmlal_3Op_v2<0b0, 0b1001, "sqdmlal",
                                           int_arm_neon_vqadds>;
defm SQDMLSLvvv : NeonI_3VDL_qdmlal_3Op_v2<0b0, 0b1011, "sqdmlsl",
                                           int_arm_neon_vqsubs>;

multiclass NeonI_3VDL_v2<bit u, bits<4> opcode,
                         string asmop, SDPatternOperator opnode,
                         bit Commutable = 0>
{
  let isCommutable = Commutable in {
    def _4s4h : NeonI_3VD_2Op<0b0, u, 0b01, opcode, asmop, "4s", "4h",
                              opnode, VPR128, VPR64, v4i32, v4i16>;
    def _2d2s : NeonI_3VD_2Op<0b0, u, 0b10, opcode, asmop, "2d", "2s",
                              opnode, VPR128, VPR64, v2i64, v2i32>;
  }
}

defm SQDMULLvvv : NeonI_3VDL_v2<0b0, 0b1101, "sqdmull",
                                int_arm_neon_vqdmull, 1>;

multiclass NeonI_3VDL2_2Op_mull_v2<bit u, bits<4> opcode,
                                   string asmop, 
                                   string opnode,
                                   bit Commutable = 0>
{
  let isCommutable = Commutable in {
    def _4s8h : NeonI_3VDL2_2Op_mull<0b1, u, 0b01, opcode, asmop, "4s", "8h",
                                     !cast<PatFrag>(opnode # "_8H"),
                                     v4i32, v8i16>;
    def _2d4s : NeonI_3VDL2_2Op_mull<0b1, u, 0b10, opcode, asmop, "2d", "4s",
                                     !cast<PatFrag>(opnode # "_4S"),
                                     v2i64, v4i32>;
  }
}

defm SQDMULL2vvv : NeonI_3VDL2_2Op_mull_v2<0b0, 0b1101, "sqdmull2", 
                                           "NI_qdmull_hi", 1>;

multiclass NeonI_3VDL2_3Op_qdmlal_v2<bit u, bits<4> opcode,
                                     string asmop, 
                                     SDPatternOperator opnode>
{
  def _4s8h : NeonI_3VDL2_3Op_mlas<0b1, u, 0b01, opcode, asmop, "4s", "8h",
                                   opnode, NI_qdmull_hi_8H,
                                   VPR128, v4i32, v8i16>;
  def _2d4s : NeonI_3VDL2_3Op_mlas<0b1, u, 0b10, opcode, asmop, "2d", "4s",
                                   opnode, NI_qdmull_hi_4S,
                                   VPR128, v2i64, v4i32>;
}

defm SQDMLAL2vvv : NeonI_3VDL2_3Op_qdmlal_v2<0b0, 0b1001, "sqdmlal2",
                                             int_arm_neon_vqadds>;
defm SQDMLSL2vvv : NeonI_3VDL2_3Op_qdmlal_v2<0b0, 0b1011, "sqdmlsl2",
                                             int_arm_neon_vqsubs>;

multiclass NeonI_3VDL_v3<bit u, bits<4> opcode,
                                   string asmop, SDPatternOperator opnode,
                                   bit Commutable = 0>
{
  let isCommutable = Commutable in {
    def _8h8b : NeonI_3VD_2Op<0b0, u, 0b00, opcode, asmop, "8h", "8b",
                              opnode, VPR128, VPR64, v8i16, v8i8>;
  }
}

defm PMULLvvv : NeonI_3VDL_v3<0b0, 0b1110, "pmull", int_arm_neon_vmullp, 1>;

multiclass NeonI_3VDL2_2Op_mull_v3<bit u, bits<4> opcode,
                                   string asmop, 
                                   string opnode,
                                   bit Commutable = 0>
{
  let isCommutable = Commutable in {
    def _8h16b : NeonI_3VDL2_2Op_mull<0b1, u, 0b00, opcode, asmop, "8h", "16b",
                                      !cast<PatFrag>(opnode # "_16B"),
                                      v8i16, v16i8>;
  }
}

defm PMULL2vvv : NeonI_3VDL2_2Op_mull_v3<0b0, 0b1110, "pmull2",
                                         "NI_pmull_hi", 1>;

// End of implementation for instruction class (3V Diff)

// The followings are vector load/store multiple N-element structure
// (class SIMD lselem).

// ld1:         load multiple 1-element structure to 1/2/3/4 registers.
// ld2/ld3/ld4: load multiple N-element structure to N registers (N = 2, 3, 4).
//              The structure consists of a sequence of sets of N values.
//              The first element of the structure is placed in the first lane
//              of the first first vector, the second element in the first lane
//              of the second vector, and so on. 
// E.g. LD1_3V_2S will load 32-bit elements {A, B, C, D, E, F} sequentially into
// the three 64-bit vectors list {BA, DC, FE}.
// E.g. LD3_2S will load 32-bit elements {A, B, C, D, E, F} into the three
// 64-bit vectors list {DA, EB, FC}.
// Store instructions store multiple structure to N registers like load.


class NeonI_LDVList<bit q, bits<4> opcode, bits<2> size,
                    RegisterOperand VecList, string asmop>
  : NeonI_LdStMult<q, 1, opcode, size,
                 (outs VecList:$Rt), (ins GPR64xsp:$Rn),
                 asmop # "\t$Rt, [$Rn]",
                 [],
                 NoItinerary> {
  let mayLoad = 1;
  let neverHasSideEffects = 1;
}

multiclass LDVList_BHSD<bits<4> opcode, string List, string asmop> {
  def _8B : NeonI_LDVList<0, opcode, 0b00,
                          !cast<RegisterOperand>(List # "8B_operand"), asmop>;

  def _4H : NeonI_LDVList<0, opcode, 0b01,
                          !cast<RegisterOperand>(List # "4H_operand"), asmop>;

  def _2S : NeonI_LDVList<0, opcode, 0b10,
                          !cast<RegisterOperand>(List # "2S_operand"), asmop>;

  def _16B : NeonI_LDVList<1, opcode, 0b00,
                           !cast<RegisterOperand>(List # "16B_operand"), asmop>;

  def _8H : NeonI_LDVList<1, opcode, 0b01,
                          !cast<RegisterOperand>(List # "8H_operand"), asmop>;

  def _4S : NeonI_LDVList<1, opcode, 0b10,
                          !cast<RegisterOperand>(List # "4S_operand"), asmop>;

  def _2D : NeonI_LDVList<1, opcode, 0b11,
                          !cast<RegisterOperand>(List # "2D_operand"), asmop>;
}

// Load multiple N-element structure to N consecutive registers (N = 1,2,3,4)
defm LD1 : LDVList_BHSD<0b0111, "VOne", "ld1">;
def LD1_1D : NeonI_LDVList<0, 0b0111, 0b11, VOne1D_operand, "ld1">;

defm LD2 : LDVList_BHSD<0b1000, "VPair", "ld2">;

defm LD3 : LDVList_BHSD<0b0100, "VTriple", "ld3">;

defm LD4 : LDVList_BHSD<0b0000, "VQuad", "ld4">;

// Load multiple 1-element structure to N consecutive registers (N = 2,3,4)
defm LD1_2V : LDVList_BHSD<0b1010, "VPair", "ld1">;
def LD1_2V_1D : NeonI_LDVList<0, 0b1010, 0b11, VPair1D_operand, "ld1">;

defm LD1_3V : LDVList_BHSD<0b0110, "VTriple", "ld1">;
def LD1_3V_1D : NeonI_LDVList<0, 0b0110, 0b11, VTriple1D_operand, "ld1">;

defm LD1_4V : LDVList_BHSD<0b0010, "VQuad", "ld1">;
def LD1_4V_1D : NeonI_LDVList<0, 0b0010, 0b11, VQuad1D_operand, "ld1">;

class NeonI_STVList<bit q, bits<4> opcode, bits<2> size,
                    RegisterOperand VecList, string asmop>
  : NeonI_LdStMult<q, 0, opcode, size,
                 (outs), (ins GPR64xsp:$Rn, VecList:$Rt), 
                 asmop # "\t$Rt, [$Rn]",
                 [], 
                 NoItinerary> {
  let mayStore = 1;
  let neverHasSideEffects = 1;
}

multiclass STVList_BHSD<bits<4> opcode, string List, string asmop> {
  def _8B : NeonI_STVList<0, opcode, 0b00,
                          !cast<RegisterOperand>(List # "8B_operand"), asmop>;

  def _4H : NeonI_STVList<0, opcode, 0b01,
                          !cast<RegisterOperand>(List # "4H_operand"), asmop>;

  def _2S : NeonI_STVList<0, opcode, 0b10,
                          !cast<RegisterOperand>(List # "2S_operand"), asmop>;

  def _16B : NeonI_STVList<1, opcode, 0b00,
                           !cast<RegisterOperand>(List # "16B_operand"), asmop>;

  def _8H : NeonI_STVList<1, opcode, 0b01,
                          !cast<RegisterOperand>(List # "8H_operand"), asmop>;

  def _4S : NeonI_STVList<1, opcode, 0b10,
                          !cast<RegisterOperand>(List # "4S_operand"), asmop>;

  def _2D : NeonI_STVList<1, opcode, 0b11,
                          !cast<RegisterOperand>(List # "2D_operand"), asmop>;
}

// Store multiple N-element structures from N registers (N = 1,2,3,4)
defm ST1 : STVList_BHSD<0b0111, "VOne", "st1">;
def ST1_1D : NeonI_STVList<0, 0b0111, 0b11, VOne1D_operand, "st1">;

defm ST2 : STVList_BHSD<0b1000, "VPair", "st2">;

defm ST3 : STVList_BHSD<0b0100, "VTriple", "st3">;

defm ST4 : STVList_BHSD<0b0000, "VQuad", "st4">;

// Store multiple 1-element structures from N consecutive registers (N = 2,3,4)
defm ST1_2V : STVList_BHSD<0b1010, "VPair", "st1">;
def ST1_2V_1D : NeonI_STVList<0, 0b1010, 0b11, VPair1D_operand, "st1">;

defm ST1_3V : STVList_BHSD<0b0110, "VTriple", "st1">;
def ST1_3V_1D : NeonI_STVList<0, 0b0110, 0b11, VTriple1D_operand, "st1">;

defm ST1_4V : STVList_BHSD<0b0010, "VQuad", "st1">;
def ST1_4V_1D : NeonI_STVList<0, 0b0010, 0b11, VQuad1D_operand, "st1">;

// End of vector load/store multiple N-element structure(class SIMD lselem)

// Scalar Three Same

class NeonI_Scalar3Same_D_size<bit u, bits<5> opcode, string asmop>
  : NeonI_Scalar3Same<u, 0b11, opcode,
                (outs FPR64:$Rd), (ins FPR64:$Rn, FPR64:$Rm),
                !strconcat(asmop, " $Rd, $Rn, $Rm"),
                [],
                NoItinerary>;

multiclass NeonI_Scalar3Same_HS_sizes<bit u, bits<5> opcode,
                                      string asmop, bit Commutable = 0>
{
  let isCommutable = Commutable in {
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
  }
}

multiclass NeonI_Scalar3Same_SD_sizes<bit u, bit size_high, bits<5> opcode,
                                      string asmop, bit Commutable = 0>
{
  let isCommutable = Commutable in {
    def sss : NeonI_Scalar3Same<u, {size_high, 0b0}, opcode,
                                (outs FPR32:$Rd), (ins FPR32:$Rn, FPR32:$Rm),
                                !strconcat(asmop, " $Rd, $Rn, $Rm"),
                                [],
                                NoItinerary>;
    def ddd : NeonI_Scalar3Same<u, {size_high, 0b1}, opcode,
                                (outs FPR64:$Rd), (ins FPR64:$Rn, FPR64:$Rm),
                                !strconcat(asmop, " $Rd, $Rn, $Rm"),
                                [],
                                NoItinerary>;
  }
}

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

multiclass Neon_Scalar3Same_D_size_patterns<SDPatternOperator opnode,
                                            Instruction INSTD> {
  def : Pat<(v1i64 (opnode (v1i64 FPR64:$Rn), (v1i64 FPR64:$Rm))),
            (INSTD FPR64:$Rn, FPR64:$Rm)>;        
}

multiclass Neon_Scalar3Same_BHSD_size_patterns<SDPatternOperator opnode,
                                               Instruction INSTB,
                                               Instruction INSTH,
                                               Instruction INSTS,
                                               Instruction INSTD>
  : Neon_Scalar3Same_D_size_patterns<opnode, INSTD> {
  def: Pat<(v1i8 (opnode (v1i8 FPR8:$Rn), (v1i8 FPR8:$Rm))),
           (INSTB FPR8:$Rn, FPR8:$Rm)>;

  def: Pat<(v1i16 (opnode (v1i16 FPR16:$Rn), (v1i16 FPR16:$Rm))),
           (INSTH FPR16:$Rn, FPR16:$Rm)>;

  def: Pat<(v1i32 (opnode (v1i32 FPR32:$Rn), (v1i32 FPR32:$Rm))),
           (INSTS FPR32:$Rn, FPR32:$Rm)>;
}

class Neon_Scalar3Same_cmp_D_size_patterns<SDPatternOperator opnode,
                                           Instruction INSTD>
  : Pat<(v1i64 (opnode (v1i64 VPR64:$Rn), (v1i64 VPR64:$Rm))),
        (INSTD VPR64:$Rn, VPR64:$Rm)>;

multiclass Neon_Scalar3Same_HS_size_patterns<SDPatternOperator opnode,
                                             Instruction INSTH,
                                             Instruction INSTS> {
  def : Pat<(v1i16 (opnode (v1i16 FPR16:$Rn), (v1i16 FPR16:$Rm))),
            (INSTH FPR16:$Rn, FPR16:$Rm)>;
  def : Pat<(v1i32 (opnode (v1i32 FPR32:$Rn), (v1i32 FPR32:$Rm))),
            (INSTS FPR32:$Rn, FPR32:$Rm)>;
}

multiclass Neon_Scalar3Same_SD_size_patterns<SDPatternOperator opnode,
                                             Instruction INSTS,
                                             Instruction INSTD> {
  def : Pat<(v1f32 (opnode (v1f32 FPR32:$Rn), (v1f32 FPR32:$Rm))),
            (INSTS FPR32:$Rn, FPR32:$Rm)>;
  def : Pat<(v1f64 (opnode (v1f64 FPR64:$Rn), (v1f64 FPR64:$Rm))),
            (INSTD FPR64:$Rn, FPR64:$Rm)>;
}

// Scalar Three Different

multiclass NeonI_Scalar3Diff_HS_size<bit u, bits<4> opcode, string asmop> {
  def shh : NeonI_Scalar3Diff<u, 0b01, opcode,
                              (outs FPR32:$Rd), (ins FPR16:$Rn, FPR16:$Rm),
                              !strconcat(asmop, " $Rd, $Rn, $Rm"),
                              [],
                              NoItinerary>;
  def dss : NeonI_Scalar3Diff<u, 0b10, opcode,
                              (outs FPR64:$Rd), (ins FPR32:$Rn, FPR32:$Rm),
                              !strconcat(asmop, " $Rd, $Rn, $Rm"),
                              [],
                              NoItinerary>;
}

multiclass NeonI_Scalar3Diff_ml_HS_size<bit u, bits<4> opcode, string asmop> {
  let Constraints = "$Src = $Rd" in {
    def shh : NeonI_Scalar3Diff<u, 0b01, opcode,
                       (outs FPR32:$Rd), (ins FPR32:$Src, FPR16:$Rn, FPR16:$Rm),
                       !strconcat(asmop, " $Rd, $Rn, $Rm"),
                       [],
                       NoItinerary>;
    def dss : NeonI_Scalar3Diff<u, 0b10, opcode,
                       (outs FPR64:$Rd), (ins FPR64:$Src, FPR32:$Rn, FPR32:$Rm),
                       !strconcat(asmop, " $Rd, $Rn, $Rm"),
                       [],
                       NoItinerary>;
  }
}

multiclass Neon_Scalar3Diff_HS_size_patterns<SDPatternOperator opnode,
                                             Instruction INSTH,
                                             Instruction INSTS> {
  def : Pat<(v1i32 (opnode (v1i16 FPR16:$Rn), (v1i16 FPR16:$Rm))),
            (INSTH FPR16:$Rn, FPR16:$Rm)>;
  def : Pat<(v1i64 (opnode (v1i32 FPR32:$Rn), (v1i32 FPR32:$Rm))),
            (INSTS FPR32:$Rn, FPR32:$Rm)>;
}

multiclass Neon_Scalar3Diff_ml_HS_size_patterns<SDPatternOperator opnode,
                                             Instruction INSTH,
                                             Instruction INSTS> {
  def : Pat<(v1i32 (opnode (v1i32 FPR32:$Src), (v1i16 FPR16:$Rn), (v1i16 FPR16:$Rm))),
            (INSTH FPR32:$Src, FPR16:$Rn, FPR16:$Rm)>;
  def : Pat<(v1i64 (opnode (v1i64 FPR64:$Src), (v1i32 FPR32:$Rn), (v1i32 FPR32:$Rm))),
            (INSTS FPR64:$Src, FPR32:$Rn, FPR32:$Rm)>;
}

// Scalar Two Registers Miscellaneous

multiclass NeonI_Scalar2SameMisc_SD_size<bit u, bit size_high, bits<5> opcode,
                                         string asmop> {
  def ss : NeonI_Scalar2SameMisc<u, {size_high, 0b0}, opcode,
                          (outs FPR32:$Rd), (ins FPR32:$Rn),
                          !strconcat(asmop, " $Rd, $Rn"),
                          [], NoItinerary>;
  def dd : NeonI_Scalar2SameMisc<u, {size_high, 0b1}, opcode,
                          (outs FPR64:$Rd), (ins FPR64:$Rn),
                          !strconcat(asmop, " $Rd, $Rn"),
                          [], NoItinerary>;
}

multiclass NeonI_Scalar2SameMisc_D_size<bit u, bits<5> opcode, string asmop> {
  def dd: NeonI_Scalar2SameMisc<u, 0b11, opcode,
                         (outs FPR64:$Rd), (ins FPR64:$Rn),
                         !strconcat(asmop, " $Rd, $Rn"),
                         [], NoItinerary>;
}

multiclass NeonI_Scalar2SameMisc_BHSD_size<bit u, bits<5> opcode, string asmop>
  : NeonI_Scalar2SameMisc_D_size<u, opcode, asmop> {
  def bb : NeonI_Scalar2SameMisc<u, 0b00, opcode,
                          (outs FPR8:$Rd), (ins FPR8:$Rn),
                          !strconcat(asmop, " $Rd, $Rn"),
                          [], NoItinerary>;
  def hh : NeonI_Scalar2SameMisc<u, 0b01, opcode,
                          (outs FPR16:$Rd), (ins FPR16:$Rn),
                          !strconcat(asmop, " $Rd, $Rn"),
                          [], NoItinerary>;
  def ss : NeonI_Scalar2SameMisc<u, 0b10, opcode,
                          (outs FPR32:$Rd), (ins FPR32:$Rn),
                          !strconcat(asmop, " $Rd, $Rn"),
                          [], NoItinerary>;
}

multiclass NeonI_Scalar2SameMisc_narrow_HSD_size<bit u, bits<5> opcode,
                                                 string asmop> {
  def bh : NeonI_Scalar2SameMisc<u, 0b00, opcode,
                          (outs FPR8:$Rd), (ins FPR16:$Rn),
                          !strconcat(asmop, " $Rd, $Rn"),
                          [], NoItinerary>;
  def hs : NeonI_Scalar2SameMisc<u, 0b01, opcode,
                          (outs FPR16:$Rd), (ins FPR32:$Rn),
                          !strconcat(asmop, " $Rd, $Rn"),
                          [], NoItinerary>;
  def sd : NeonI_Scalar2SameMisc<u, 0b10, opcode,
                          (outs FPR32:$Rd), (ins FPR64:$Rn),
                          !strconcat(asmop, " $Rd, $Rn"),
                          [], NoItinerary>;
}

multiclass NeonI_Scalar2SameMisc_accum_BHSD_size<bit u, bits<5> opcode,
                                                 string asmop> {

  let Constraints = "$Src = $Rd" in {
    def bb : NeonI_Scalar2SameMisc<u, 0b00, opcode,
                            (outs FPR8:$Rd), (ins FPR8:$Src, FPR8:$Rn),
                            !strconcat(asmop, " $Rd, $Rn"),
                            [], NoItinerary>;
    def hh : NeonI_Scalar2SameMisc<u, 0b01, opcode,
                            (outs FPR16:$Rd), (ins FPR16:$Src, FPR16:$Rn),
                            !strconcat(asmop, " $Rd, $Rn"),
                            [], NoItinerary>;
    def ss : NeonI_Scalar2SameMisc<u, 0b10, opcode,
                            (outs FPR32:$Rd), (ins FPR32:$Src, FPR32:$Rn),
                            !strconcat(asmop, " $Rd, $Rn"),
                            [], NoItinerary>;
    def dd: NeonI_Scalar2SameMisc<u, 0b11, opcode,
                           (outs FPR64:$Rd), (ins FPR64:$Src, FPR64:$Rn),
                           !strconcat(asmop, " $Rd, $Rn"),
                           [], NoItinerary>;
  }
}

multiclass Neon_Scalar2SameMisc_cvt_SD_size_patterns<SDPatternOperator Sopnode,
                                                     SDPatternOperator Dopnode,
                                                     Instruction INSTS,
                                                     Instruction INSTD> {
  def : Pat<(v1f32 (Sopnode (v1i32 FPR32:$Rn))),
            (INSTS FPR32:$Rn)>;
  def : Pat<(v1f64 (Dopnode (v1i64 FPR64:$Rn))),
            (INSTD FPR64:$Rn)>;
}

multiclass Neon_Scalar2SameMisc_SD_size_patterns<SDPatternOperator opnode,
                                                 Instruction INSTS,
                                                 Instruction INSTD> {
  def : Pat<(v1f32 (opnode (v1f32 FPR32:$Rn))),
            (INSTS FPR32:$Rn)>;
  def : Pat<(v1f64 (opnode (v1f64 FPR64:$Rn))),
            (INSTD FPR64:$Rn)>;
}

class NeonI_Scalar2SameMisc_cmpz_D_size<bit u, bits<5> opcode, string asmop>
  : NeonI_Scalar2SameMisc<u, 0b11, opcode,
                          (outs FPR64:$Rd), (ins FPR64:$Rn, neon_uimm0:$Imm),
                          !strconcat(asmop, " $Rd, $Rn, $Imm"),
                          [],
                          NoItinerary>;

class Neon_Scalar2SameMisc_cmpz_D_size_patterns<SDPatternOperator opnode,
                                                Instruction INSTD>
  : Pat<(v1i64 (opnode (v1i64 VPR64:$Rn), (v1i64 (bitconvert (v8i8 Neon_immAllZeros))))),
        (INSTD VPR64:$Rn, 0)>;

multiclass Neon_Scalar2SameMisc_D_size_patterns<SDPatternOperator opnode,
                                                Instruction INSTD> {
  def : Pat<(v1i64 (opnode (v1i64 FPR64:$Rn))),
            (INSTD FPR64:$Rn)>;
}

multiclass Neon_Scalar2SameMisc_BHSD_size_patterns<SDPatternOperator opnode,
                                                   Instruction INSTB,
                                                   Instruction INSTH,
                                                   Instruction INSTS,
                                                   Instruction INSTD>
  : Neon_Scalar2SameMisc_D_size_patterns<opnode, INSTD> {
  def : Pat<(v1i8 (opnode (v1i8 FPR8:$Rn))),
            (INSTB FPR8:$Rn)>;
  def : Pat<(v1i16 (opnode (v1i16 FPR16:$Rn))),
            (INSTH FPR16:$Rn)>;
  def : Pat<(v1i32 (opnode (v1i32 FPR32:$Rn))),
            (INSTS FPR32:$Rn)>;
}

multiclass Neon_Scalar2SameMisc_narrow_HSD_size_patterns<
                                                       SDPatternOperator opnode,
                                                       Instruction INSTH,
                                                       Instruction INSTS,
                                                       Instruction INSTD> {
  def : Pat<(v1i8 (opnode (v1i16 FPR16:$Rn))),
            (INSTH FPR16:$Rn)>;
  def : Pat<(v1i16 (opnode (v1i32 FPR32:$Rn))),
            (INSTS FPR32:$Rn)>;
  def : Pat<(v1i32 (opnode (v1i64 FPR64:$Rn))),
            (INSTD FPR64:$Rn)>;

}

multiclass Neon_Scalar2SameMisc_accum_BHSD_size_patterns<
                                                       SDPatternOperator opnode,
                                                       Instruction INSTB,
                                                       Instruction INSTH,
                                                       Instruction INSTS,
                                                       Instruction INSTD> {
  def : Pat<(v1i8 (opnode (v1i8 FPR8:$Src), (v1i8 FPR8:$Rn))),
            (INSTB FPR8:$Src, FPR8:$Rn)>;
  def : Pat<(v1i16 (opnode (v1i16 FPR16:$Src), (v1i16 FPR16:$Rn))),
            (INSTH FPR16:$Src, FPR16:$Rn)>;
  def : Pat<(v1i32 (opnode (v1i32 FPR32:$Src), (v1i32 FPR32:$Rn))),
            (INSTS FPR32:$Src, FPR32:$Rn)>;
  def : Pat<(v1i64 (opnode (v1i64 FPR64:$Src), (v1i64 FPR64:$Rn))),
            (INSTD FPR64:$Src, FPR64:$Rn)>;
}

// Scalar Integer Add
let isCommutable = 1 in {
def ADDddd : NeonI_Scalar3Same_D_size<0b0, 0b10000, "add">;
}

// Scalar Integer Sub
def SUBddd : NeonI_Scalar3Same_D_size<0b1, 0b10000, "sub">;

// Pattern for Scalar Integer Add and Sub with D register only
defm : Neon_Scalar3Same_D_size_patterns<add, ADDddd>;
defm : Neon_Scalar3Same_D_size_patterns<sub, SUBddd>;

// Patterns to match llvm.aarch64.* intrinsic for Scalar Add, Sub
defm : Neon_Scalar3Same_D_size_patterns<int_aarch64_neon_vaddds, ADDddd>;
defm : Neon_Scalar3Same_D_size_patterns<int_aarch64_neon_vadddu, ADDddd>;
defm : Neon_Scalar3Same_D_size_patterns<int_aarch64_neon_vsubds, SUBddd>;
defm : Neon_Scalar3Same_D_size_patterns<int_aarch64_neon_vsubdu, SUBddd>;

// Scalar Integer Saturating Add (Signed, Unsigned)
defm SQADD : NeonI_Scalar3Same_BHSD_sizes<0b0, 0b00001, "sqadd", 1>;
defm UQADD : NeonI_Scalar3Same_BHSD_sizes<0b1, 0b00001, "uqadd", 1>;

// Scalar Integer Saturating Sub (Signed, Unsigned)
defm SQSUB : NeonI_Scalar3Same_BHSD_sizes<0b0, 0b00101, "sqsub", 0>;
defm UQSUB : NeonI_Scalar3Same_BHSD_sizes<0b1, 0b00101, "uqsub", 0>;

// Patterns to match llvm.arm.* intrinsic for
// Scalar Integer Saturating Add, Sub  (Signed, Unsigned)
defm : Neon_Scalar3Same_D_size_patterns<int_arm_neon_vqadds, SQADDddd>;
defm : Neon_Scalar3Same_D_size_patterns<int_arm_neon_vqaddu, UQADDddd>;
defm : Neon_Scalar3Same_D_size_patterns<int_arm_neon_vqsubs, SQSUBddd>;
defm : Neon_Scalar3Same_D_size_patterns<int_arm_neon_vqsubu, UQSUBddd>;

// Patterns to match llvm.aarch64.* intrinsic for
// Scalar Integer Saturating Add, Sub  (Signed, Unsigned)
defm : Neon_Scalar3Same_BHSD_size_patterns<int_aarch64_neon_vqadds, SQADDbbb,
                                           SQADDhhh, SQADDsss, SQADDddd>;
defm : Neon_Scalar3Same_BHSD_size_patterns<int_aarch64_neon_vqaddu, UQADDbbb,
                                           UQADDhhh, UQADDsss, UQADDddd>;
defm : Neon_Scalar3Same_BHSD_size_patterns<int_aarch64_neon_vqsubs, SQSUBbbb,
                                           SQSUBhhh, SQSUBsss, SQSUBddd>;
defm : Neon_Scalar3Same_BHSD_size_patterns<int_aarch64_neon_vqsubu, UQSUBbbb,
                                           UQSUBhhh, UQSUBsss, UQSUBddd>;

// Scalar Integer Saturating Doubling Multiply Half High
defm SQDMULH : NeonI_Scalar3Same_HS_sizes<0b0, 0b10110, "sqdmulh", 1>;

// Scalar Integer Saturating Rounding Doubling Multiply Half High
defm SQRDMULH : NeonI_Scalar3Same_HS_sizes<0b1, 0b10110, "sqrdmulh", 1>;

// Patterns to match llvm.arm.* intrinsic for
// Scalar Integer Saturating Doubling Multiply Half High and
// Scalar Integer Saturating Rounding Doubling Multiply Half High
defm : Neon_Scalar3Same_HS_size_patterns<int_arm_neon_vqdmulh, SQDMULHhhh,
                                                               SQDMULHsss>;
defm : Neon_Scalar3Same_HS_size_patterns<int_arm_neon_vqrdmulh, SQRDMULHhhh,
                                                                SQRDMULHsss>;

// Scalar Floating-point Multiply Extended
defm FMULX : NeonI_Scalar3Same_SD_sizes<0b0, 0b0, 0b11011, "fmulx", 1>;

// Scalar Floating-point Reciprocal Step
defm FRECPS : NeonI_Scalar3Same_SD_sizes<0b0, 0b0, 0b11111, "frecps", 0>;

// Scalar Floating-point Reciprocal Square Root Step
defm FRSQRTS : NeonI_Scalar3Same_SD_sizes<0b0, 0b1, 0b11111, "frsqrts", 0>;

// Patterns to match llvm.arm.* intrinsic for
// Scalar Floating-point Reciprocal Step and
// Scalar Floating-point Reciprocal Square Root Step
defm : Neon_Scalar3Same_SD_size_patterns<int_arm_neon_vrecps, FRECPSsss,
                                                              FRECPSddd>;
defm : Neon_Scalar3Same_SD_size_patterns<int_arm_neon_vrsqrts, FRSQRTSsss,
                                                               FRSQRTSddd>;

// Patterns to match llvm.aarch64.* intrinsic for
// Scalar Floating-point Multiply Extended,
defm : Neon_Scalar3Same_SD_size_patterns<int_aarch64_neon_vmulx, FMULXsss,
                                         FMULXddd>;

// Scalar Integer Shift Left (Signed, Unsigned)
def SSHLddd : NeonI_Scalar3Same_D_size<0b0, 0b01000, "sshl">;
def USHLddd : NeonI_Scalar3Same_D_size<0b1, 0b01000, "ushl">;

// Patterns to match llvm.arm.* intrinsic for
// Scalar Integer Shift Left (Signed, Unsigned)
defm : Neon_Scalar3Same_D_size_patterns<int_arm_neon_vshifts, SSHLddd>;
defm : Neon_Scalar3Same_D_size_patterns<int_arm_neon_vshiftu, USHLddd>;

// Patterns to match llvm.aarch64.* intrinsic for
// Scalar Integer Shift Left (Signed, Unsigned)
defm : Neon_Scalar3Same_D_size_patterns<int_aarch64_neon_vshlds, SSHLddd>;
defm : Neon_Scalar3Same_D_size_patterns<int_aarch64_neon_vshldu, USHLddd>;

// Scalar Integer Saturating Shift Left (Signed, Unsigned)
defm SQSHL: NeonI_Scalar3Same_BHSD_sizes<0b0, 0b01001, "sqshl", 0>;
defm UQSHL: NeonI_Scalar3Same_BHSD_sizes<0b1, 0b01001, "uqshl", 0>;

// Patterns to match llvm.aarch64.* intrinsic for
// Scalar  Integer Saturating Shift Letf (Signed, Unsigned)
defm : Neon_Scalar3Same_BHSD_size_patterns<int_aarch64_neon_vqshls, SQSHLbbb,
                                           SQSHLhhh, SQSHLsss, SQSHLddd>;
defm : Neon_Scalar3Same_BHSD_size_patterns<int_aarch64_neon_vqshlu, UQSHLbbb,
                                           UQSHLhhh, UQSHLsss, UQSHLddd>;

// Patterns to match llvm.arm.* intrinsic for
// Scalar  Integer Saturating Shift Letf (Signed, Unsigned)
defm : Neon_Scalar3Same_D_size_patterns<int_arm_neon_vqshifts, SQSHLddd>;
defm : Neon_Scalar3Same_D_size_patterns<int_arm_neon_vqshiftu, UQSHLddd>;

// Scalar Integer Rounding Shift Left (Signed, Unsigned)
def SRSHLddd: NeonI_Scalar3Same_D_size<0b0, 0b01010, "srshl">;
def URSHLddd: NeonI_Scalar3Same_D_size<0b1, 0b01010, "urshl">;

// Patterns to match llvm.aarch64.* intrinsic for
// Scalar Integer Rounding Shift Left (Signed, Unsigned)
defm : Neon_Scalar3Same_D_size_patterns<int_aarch64_neon_vrshlds, SRSHLddd>;
defm : Neon_Scalar3Same_D_size_patterns<int_aarch64_neon_vrshldu, URSHLddd>;

// Patterns to match llvm.arm.* intrinsic for
// Scalar Integer Rounding Shift Left (Signed, Unsigned)
defm : Neon_Scalar3Same_D_size_patterns<int_arm_neon_vrshifts, SRSHLddd>;
defm : Neon_Scalar3Same_D_size_patterns<int_arm_neon_vrshiftu, URSHLddd>;

// Scalar Integer Saturating Rounding Shift Left (Signed, Unsigned)
defm SQRSHL: NeonI_Scalar3Same_BHSD_sizes<0b0, 0b01011, "sqrshl", 0>;
defm UQRSHL: NeonI_Scalar3Same_BHSD_sizes<0b1, 0b01011, "uqrshl", 0>;

// Patterns to match llvm.aarch64.* intrinsic for
// Scalar Integer Saturating Rounding Shift Left (Signed, Unsigned)
defm : Neon_Scalar3Same_BHSD_size_patterns<int_aarch64_neon_vqrshls, SQRSHLbbb,
                                           SQRSHLhhh, SQRSHLsss, SQRSHLddd>;
defm : Neon_Scalar3Same_BHSD_size_patterns<int_aarch64_neon_vqrshlu, UQRSHLbbb,
                                           UQRSHLhhh, UQRSHLsss, UQRSHLddd>;

// Patterns to match llvm.arm.* intrinsic for
// Scalar Integer Saturating Rounding Shift Left (Signed, Unsigned)
defm : Neon_Scalar3Same_D_size_patterns<int_arm_neon_vqrshifts, SQRSHLddd>;
defm : Neon_Scalar3Same_D_size_patterns<int_arm_neon_vqrshiftu, UQRSHLddd>;

// Signed Saturating Doubling Multiply-Add Long
defm SQDMLAL : NeonI_Scalar3Diff_ml_HS_size<0b0, 0b1001, "sqdmlal">;
defm : Neon_Scalar3Diff_ml_HS_size_patterns<int_aarch64_neon_vqdmlal,
                                            SQDMLALshh, SQDMLALdss>;

// Signed Saturating Doubling Multiply-Subtract Long
defm SQDMLSL : NeonI_Scalar3Diff_ml_HS_size<0b0, 0b1011, "sqdmlsl">;
defm : Neon_Scalar3Diff_ml_HS_size_patterns<int_aarch64_neon_vqdmlsl,
                                            SQDMLSLshh, SQDMLSLdss>;

// Signed Saturating Doubling Multiply Long
defm SQDMULL : NeonI_Scalar3Diff_HS_size<0b0, 0b1101, "sqdmull">;
defm : Neon_Scalar3Diff_HS_size_patterns<int_aarch64_neon_vqdmull,
                                         SQDMULLshh, SQDMULLdss>;

// Scalar Signed Integer Convert To Floating-point
defm SCVTF  : NeonI_Scalar2SameMisc_SD_size<0b0, 0b0, 0b11101, "scvtf">;
defm : Neon_Scalar2SameMisc_cvt_SD_size_patterns<int_aarch64_neon_vcvtf32_s32,
                                                 int_aarch64_neon_vcvtf64_s64,
                                                 SCVTFss, SCVTFdd>;

// Scalar Unsigned Integer Convert To Floating-point
defm UCVTF  : NeonI_Scalar2SameMisc_SD_size<0b1, 0b0, 0b11101, "ucvtf">;
defm : Neon_Scalar2SameMisc_cvt_SD_size_patterns<int_aarch64_neon_vcvtf32_u32,
                                                 int_aarch64_neon_vcvtf64_u64,
                                                 UCVTFss, UCVTFdd>;

// Scalar Floating-point Reciprocal Estimate
defm FRECPE : NeonI_Scalar2SameMisc_SD_size<0b0, 0b1, 0b11101, "frecpe">;
defm : Neon_Scalar2SameMisc_SD_size_patterns<int_arm_neon_vrecpe,
                                             FRECPEss, FRECPEdd>;

// Scalar Floating-point Reciprocal Exponent
defm FRECPX : NeonI_Scalar2SameMisc_SD_size<0b0, 0b1, 0b11111, "frecpx">;
defm : Neon_Scalar2SameMisc_SD_size_patterns<int_aarch64_neon_vrecpx,
                                             FRECPXss, FRECPXdd>;

// Scalar Floating-point Reciprocal Square Root Estimate
defm FRSQRTE: NeonI_Scalar2SameMisc_SD_size<0b1, 0b1, 0b11101, "frsqrte">;
defm : Neon_Scalar2SameMisc_SD_size_patterns<int_arm_neon_vrsqrte,
                                             FRSQRTEss, FRSQRTEdd>;

// Scalar Integer Compare

// Scalar Compare Bitwise Equal
def CMEQddd: NeonI_Scalar3Same_D_size<0b1, 0b10001, "cmeq">;
def : Neon_Scalar3Same_cmp_D_size_patterns<int_aarch64_neon_vceq, CMEQddd>;

// Scalar Compare Signed Greather Than Or Equal
def CMGEddd: NeonI_Scalar3Same_D_size<0b0, 0b00111, "cmge">;
def : Neon_Scalar3Same_cmp_D_size_patterns<int_aarch64_neon_vcge, CMGEddd>;

// Scalar Compare Unsigned Higher Or Same
def CMHSddd: NeonI_Scalar3Same_D_size<0b1, 0b00111, "cmhs">;
def : Neon_Scalar3Same_cmp_D_size_patterns<int_aarch64_neon_vchs, CMHSddd>;

// Scalar Compare Unsigned Higher
def CMHIddd: NeonI_Scalar3Same_D_size<0b1, 0b00110, "cmhi">;
def : Neon_Scalar3Same_cmp_D_size_patterns<int_aarch64_neon_vchi, CMHIddd>;

// Scalar Compare Signed Greater Than
def CMGTddd: NeonI_Scalar3Same_D_size<0b0, 0b00110, "cmgt">;
def : Neon_Scalar3Same_cmp_D_size_patterns<int_aarch64_neon_vcgt, CMGTddd>;

// Scalar Compare Bitwise Test Bits
def CMTSTddd: NeonI_Scalar3Same_D_size<0b0, 0b10001, "cmtst">;
def : Neon_Scalar3Same_cmp_D_size_patterns<int_aarch64_neon_vtstd, CMTSTddd>;

// Scalar Compare Bitwise Equal To Zero
def CMEQddi: NeonI_Scalar2SameMisc_cmpz_D_size<0b0, 0b01001, "cmeq">;
def : Neon_Scalar2SameMisc_cmpz_D_size_patterns<int_aarch64_neon_vceq,
                                                CMEQddi>;

// Scalar Compare Signed Greather Than Or Equal To Zero
def CMGEddi: NeonI_Scalar2SameMisc_cmpz_D_size<0b1, 0b01000, "cmge">;
def : Neon_Scalar2SameMisc_cmpz_D_size_patterns<int_aarch64_neon_vcge,
                                                CMGEddi>;

// Scalar Compare Signed Greater Than Zero
def CMGTddi: NeonI_Scalar2SameMisc_cmpz_D_size<0b0, 0b01000, "cmgt">;
def : Neon_Scalar2SameMisc_cmpz_D_size_patterns<int_aarch64_neon_vcgt,
                                                CMGTddi>;

// Scalar Compare Signed Less Than Or Equal To Zero
def CMLEddi: NeonI_Scalar2SameMisc_cmpz_D_size<0b1, 0b01001, "cmle">;
def : Neon_Scalar2SameMisc_cmpz_D_size_patterns<int_aarch64_neon_vclez,
                                                CMLEddi>;

// Scalar Compare Less Than Zero
def CMLTddi: NeonI_Scalar2SameMisc_cmpz_D_size<0b0, 0b01010, "cmlt">;
def : Neon_Scalar2SameMisc_cmpz_D_size_patterns<int_aarch64_neon_vcltz,
                                                CMLTddi>;

// Scalar Absolute Value
defm ABS : NeonI_Scalar2SameMisc_D_size<0b0, 0b01011, "abs">;
defm : Neon_Scalar2SameMisc_D_size_patterns<int_aarch64_neon_vabs, ABSdd>;

// Scalar Signed Saturating Absolute Value
defm SQABS : NeonI_Scalar2SameMisc_BHSD_size<0b0, 0b00111, "sqabs">;
defm : Neon_Scalar2SameMisc_BHSD_size_patterns<int_arm_neon_vqabs,
                                               SQABSbb, SQABShh, SQABSss, SQABSdd>;

// Scalar Negate
defm NEG : NeonI_Scalar2SameMisc_D_size<0b1, 0b01011, "neg">;
defm : Neon_Scalar2SameMisc_D_size_patterns<int_aarch64_neon_vneg, NEGdd>;

// Scalar Signed Saturating Negate
defm SQNEG : NeonI_Scalar2SameMisc_BHSD_size<0b1, 0b00111, "sqneg">;
defm : Neon_Scalar2SameMisc_BHSD_size_patterns<int_arm_neon_vqneg,
                                               SQNEGbb, SQNEGhh, SQNEGss, SQNEGdd>;

// Scalar Signed Saturating Accumulated of Unsigned Value
defm SUQADD : NeonI_Scalar2SameMisc_accum_BHSD_size<0b0, 0b00011, "suqadd">;
defm : Neon_Scalar2SameMisc_accum_BHSD_size_patterns<int_aarch64_neon_vuqadd,
                                                     SUQADDbb, SUQADDhh,
                                                     SUQADDss, SUQADDdd>;

// Scalar Unsigned Saturating Accumulated of Signed Value
defm USQADD : NeonI_Scalar2SameMisc_accum_BHSD_size<0b1, 0b00011, "usqadd">;
defm : Neon_Scalar2SameMisc_accum_BHSD_size_patterns<int_aarch64_neon_vsqadd,
                                                     USQADDbb, USQADDhh,
                                                     USQADDss, USQADDdd>;

// Scalar Signed Saturating Extract Unsigned Narrow
defm SQXTUN : NeonI_Scalar2SameMisc_narrow_HSD_size<0b1, 0b10010, "sqxtun">;
defm : Neon_Scalar2SameMisc_narrow_HSD_size_patterns<int_arm_neon_vqmovnsu,
                                                     SQXTUNbh, SQXTUNhs,
                                                     SQXTUNsd>;

// Scalar Signed Saturating Extract Narrow
defm SQXTN  : NeonI_Scalar2SameMisc_narrow_HSD_size<0b0, 0b10100, "sqxtn">;
defm : Neon_Scalar2SameMisc_narrow_HSD_size_patterns<int_arm_neon_vqmovns,
                                                     SQXTNbh, SQXTNhs,
                                                     SQXTNsd>;

// Scalar Unsigned Saturating Extract Narrow
defm UQXTN  : NeonI_Scalar2SameMisc_narrow_HSD_size<0b1, 0b10100, "uqxtn">;
defm : Neon_Scalar2SameMisc_narrow_HSD_size_patterns<int_arm_neon_vqmovnu,
                                                     UQXTNbh, UQXTNhs,
                                                     UQXTNsd>;

// Scalar Reduce Pairwise

multiclass NeonI_ScalarPair_D_sizes<bit u, bit size, bits<5> opcode,
                                     string asmop, bit Commutable = 0> {
  let isCommutable = Commutable in {
    def _D_2D : NeonI_ScalarPair<u, {size, 0b1}, opcode,
                                (outs FPR64:$Rd), (ins VPR128:$Rn),
                                !strconcat(asmop, " $Rd, $Rn.2d"),
                                [],
                                NoItinerary>;
  }
}

multiclass NeonI_ScalarPair_SD_sizes<bit u, bit size, bits<5> opcode,
                                     string asmop, bit Commutable = 0>
  : NeonI_ScalarPair_D_sizes<u, size, opcode, asmop, Commutable> {
  let isCommutable = Commutable in {
    def _S_2S : NeonI_ScalarPair<u, {size, 0b0}, opcode,
                                (outs FPR32:$Rd), (ins VPR64:$Rn),
                                !strconcat(asmop, " $Rd, $Rn.2s"),
                                [],
                                NoItinerary>;
  }
}

// Scalar Reduce Addition Pairwise (Integer) with
// Pattern to match llvm.arm.* intrinsic
defm ADDPvv : NeonI_ScalarPair_D_sizes<0b0, 0b1, 0b11011, "addp", 0>;

// Pattern to match llvm.aarch64.* intrinsic for
// Scalar Reduce Addition Pairwise (Integer)
def : Pat<(v1i64 (int_aarch64_neon_vpadd (v2i64 VPR128:$Rn))),
          (ADDPvv_D_2D VPR128:$Rn)>;

// Scalar Reduce Addition Pairwise (Floating Point)
defm FADDPvv : NeonI_ScalarPair_SD_sizes<0b1, 0b0, 0b01101, "faddp", 0>;

// Scalar Reduce Maximum Pairwise (Floating Point)
defm FMAXPvv : NeonI_ScalarPair_SD_sizes<0b1, 0b0, 0b01111, "fmaxp", 0>;

// Scalar Reduce Minimum Pairwise (Floating Point)
defm FMINPvv : NeonI_ScalarPair_SD_sizes<0b1, 0b1, 0b01111, "fminp", 0>;

// Scalar Reduce maxNum Pairwise (Floating Point)
defm FMAXNMPvv : NeonI_ScalarPair_SD_sizes<0b1, 0b0, 0b01100, "fmaxnmp", 0>;

// Scalar Reduce minNum Pairwise (Floating Point)
defm FMINNMPvv : NeonI_ScalarPair_SD_sizes<0b1, 0b1, 0b01100, "fminnmp", 0>;

multiclass Neon_ScalarPair_SD_size_patterns<SDPatternOperator opnodeS,
                                            SDPatternOperator opnodeD,
                                            Instruction INSTS,
                                            Instruction INSTD> {
  def : Pat<(v1f32 (opnodeS (v2f32 VPR64:$Rn))),
            (INSTS VPR64:$Rn)>;
  def : Pat<(v1f64 (opnodeD (v2f64 VPR128:$Rn))),
            (INSTD VPR128:$Rn)>;
}

// Patterns to match llvm.aarch64.* intrinsic for
// Scalar Reduce Add, Max, Min, MaxiNum, MinNum Pairwise (Floating Point)
defm : Neon_ScalarPair_SD_size_patterns<int_aarch64_neon_vpfadd,
  int_aarch64_neon_vpfaddq, FADDPvv_S_2S, FADDPvv_D_2D>;

defm : Neon_ScalarPair_SD_size_patterns<int_aarch64_neon_vpmax,
  int_aarch64_neon_vpmaxq, FMAXPvv_S_2S, FMAXPvv_D_2D>;

defm : Neon_ScalarPair_SD_size_patterns<int_aarch64_neon_vpmin,
  int_aarch64_neon_vpminq, FMINPvv_S_2S, FMINPvv_D_2D>;

defm : Neon_ScalarPair_SD_size_patterns<int_aarch64_neon_vpfmaxnm,
  int_aarch64_neon_vpfmaxnmq, FMAXNMPvv_S_2S, FMAXNMPvv_D_2D>;

defm : Neon_ScalarPair_SD_size_patterns<int_aarch64_neon_vpfminnm, 
  int_aarch64_neon_vpfminnmq, FMINNMPvv_S_2S, FMINNMPvv_D_2D>;



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
def : Pat<(f16 (bitconvert (v1i16  FPR16:$src))), (f16 FPR16:$src)>;
def : Pat<(f32 (bitconvert (v1i32  FPR32:$src))), (f32 FPR32:$src)>;
def : Pat<(f64 (bitconvert (v1i64  FPR64:$src))), (f64 FPR64:$src)>;
def : Pat<(f32 (bitconvert (v1f32  FPR32:$src))), (f32 FPR32:$src)>;
def : Pat<(f64 (bitconvert (v1f64  FPR64:$src))), (f64 FPR64:$src)>;

def : Pat<(i64 (bitconvert (v1i64  FPR64:$src))), (FMOVxd $src)>;
def : Pat<(i32 (bitconvert (v1i32  FPR32:$src))), (FMOVws $src)>;

def : Pat<(v8i8  (bitconvert (v1i64  VPR64:$src))), (v8i8 VPR64:$src)>;
def : Pat<(v4i16 (bitconvert (v1i64  VPR64:$src))), (v4i16 VPR64:$src)>;
def : Pat<(v2i32 (bitconvert (v1i64  VPR64:$src))), (v2i32 VPR64:$src)>;

def : Pat<(f64   (bitconvert (v8i8  VPR64:$src))), (f64 VPR64:$src)>;
def : Pat<(f64   (bitconvert (v4i16  VPR64:$src))), (f64 VPR64:$src)>;
def : Pat<(f64   (bitconvert (v2i32  VPR64:$src))), (f64 VPR64:$src)>;
def : Pat<(f64   (bitconvert (v2f32  VPR64:$src))), (f64 VPR64:$src)>;
def : Pat<(f64   (bitconvert (v1i64  VPR64:$src))), (f64 VPR64:$src)>;

def : Pat<(f128  (bitconvert (v16i8  VPR128:$src))), (f128 VPR128:$src)>;
def : Pat<(f128  (bitconvert (v8i16  VPR128:$src))), (f128 VPR128:$src)>;
def : Pat<(f128  (bitconvert (v4i32  VPR128:$src))), (f128 VPR128:$src)>;
def : Pat<(f128  (bitconvert (v2i64  VPR128:$src))), (f128 VPR128:$src)>;
def : Pat<(f128  (bitconvert (v4f32  VPR128:$src))), (f128 VPR128:$src)>;
def : Pat<(f128  (bitconvert (v2f64  VPR128:$src))), (f128 VPR128:$src)>;

def : Pat<(v1i16 (bitconvert (f16  FPR16:$src))), (v1i16 FPR16:$src)>;
def : Pat<(v1i32 (bitconvert (f32  FPR32:$src))), (v1i32 FPR32:$src)>;
def : Pat<(v1i64 (bitconvert (f64  FPR64:$src))), (v1i64 FPR64:$src)>;
def : Pat<(v1f32 (bitconvert (f32  FPR32:$src))), (v1f32 FPR32:$src)>;
def : Pat<(v1f64 (bitconvert (f64  FPR64:$src))), (v1f64 FPR64:$src)>;

def : Pat<(v1i64 (bitconvert (i64  GPR64:$src))), (FMOVdx $src)>;
def : Pat<(v1i32 (bitconvert (i32  GPR32:$src))), (FMOVsw $src)>;

def : Pat<(v8i8   (bitconvert (f64   FPR64:$src))), (v8i8 FPR64:$src)>;
def : Pat<(v4i16  (bitconvert (f64   FPR64:$src))), (v4i16 FPR64:$src)>;
def : Pat<(v2i32  (bitconvert (f64   FPR64:$src))), (v2i32 FPR64:$src)>;
def : Pat<(v2f32  (bitconvert (f64   FPR64:$src))), (v2f32 FPR64:$src)>;
def : Pat<(v1i64  (bitconvert (f64   FPR64:$src))), (v1i64 FPR64:$src)>;

def : Pat<(v16i8  (bitconvert (f128   FPR128:$src))), (v16i8 FPR128:$src)>;
def : Pat<(v8i16  (bitconvert (f128   FPR128:$src))), (v8i16 FPR128:$src)>;
def : Pat<(v4i32  (bitconvert (f128   FPR128:$src))), (v4i32 FPR128:$src)>;
def : Pat<(v2i64  (bitconvert (f128   FPR128:$src))), (v2i64 FPR128:$src)>;
def : Pat<(v4f32  (bitconvert (f128   FPR128:$src))), (v4f32 FPR128:$src)>;
def : Pat<(v2f64  (bitconvert (f128   FPR128:$src))), (v2f64 FPR128:$src)>;

def neon_uimm0_bare : Operand<i64>,
                        ImmLeaf<i64, [{return Imm == 0;}]> {
  let ParserMatchClass = neon_uimm0_asmoperand;
  let PrintMethod = "printNeonUImm8OperandBare";
}

def neon_uimm1_bare : Operand<i64>,
                        ImmLeaf<i64, [{(void)Imm; return true;}]> {
  let ParserMatchClass = neon_uimm1_asmoperand;
  let PrintMethod = "printNeonUImm8OperandBare";
}

def neon_uimm2_bare : Operand<i64>,
                        ImmLeaf<i64, [{(void)Imm; return true;}]> {
  let ParserMatchClass = neon_uimm2_asmoperand;
  let PrintMethod = "printNeonUImm8OperandBare";
}

def neon_uimm3_bare : Operand<i64>,
                        ImmLeaf<i64, [{(void)Imm; return true;}]> {
  let ParserMatchClass = uimm3_asmoperand;
  let PrintMethod = "printNeonUImm8OperandBare";
}

def neon_uimm4_bare : Operand<i64>,
                        ImmLeaf<i64, [{(void)Imm; return true;}]> {
  let ParserMatchClass = uimm4_asmoperand;
  let PrintMethod = "printNeonUImm8OperandBare";
}

class NeonI_INS_main<string asmop, string Res, ValueType ResTy,
                     RegisterClass OpGPR, ValueType OpTy, Operand OpImm>
  : NeonI_copy<0b1, 0b0, 0b0011,
               (outs VPR128:$Rd), (ins VPR128:$src, OpGPR:$Rn, OpImm:$Imm),
               asmop # "\t$Rd." # Res # "[$Imm], $Rn",
               [(set (ResTy VPR128:$Rd),
                 (ResTy (vector_insert
                   (ResTy VPR128:$src),
                   (OpTy OpGPR:$Rn),
                   (OpImm:$Imm))))],
               NoItinerary> {
  bits<4> Imm;
  let Constraints = "$src = $Rd";
}

// The followings are for instruction class (3V Elem)

// Variant 1

class NI_2VE<bit q, bit u, bits<2> size, bits<4> opcode,
             string asmop, string ResS, string OpS, string EleOpS,
             Operand OpImm, RegisterOperand ResVPR,
             RegisterOperand OpVPR, RegisterOperand EleOpVPR>
  : NeonI_2VElem<q, u, size, opcode, 
                 (outs ResVPR:$Rd), (ins ResVPR:$src, OpVPR:$Rn,
                                         EleOpVPR:$Re, OpImm:$Index),
                 asmop # "\t$Rd." # ResS # ", $Rn." # OpS #
                 ", $Re." # EleOpS # "[$Index]",
                 [],
                 NoItinerary> {
  bits<3> Index;
  bits<5> Re;

  let Constraints = "$src = $Rd";
}

multiclass NI_2VE_v1<bit u, bits<4> opcode, string asmop>
{
  // vector register class for element is always 128-bit to cover the max index
  def _2s4s : NI_2VE<0b0, u, 0b10, opcode, asmop, "2s", "2s", "s",
                     neon_uimm2_bare, VPR64, VPR64, VPR128> {
    let Inst{11} = {Index{1}};
    let Inst{21} = {Index{0}};
    let Inst{20-16} = Re;
  }

  def _4s4s : NI_2VE<0b1, u, 0b10, opcode, asmop, "4s", "4s", "s",
                     neon_uimm2_bare, VPR128, VPR128, VPR128> {
    let Inst{11} = {Index{1}};
    let Inst{21} = {Index{0}};
    let Inst{20-16} = Re;
  }

  // Index operations on 16-bit(H) elements are restricted to using v0-v15.
  def _4h8h : NI_2VE<0b0, u, 0b01, opcode, asmop, "4h", "4h", "h",
                     neon_uimm3_bare, VPR64, VPR64, VPR128Lo> {
    let Inst{11} = {Index{2}};
    let Inst{21} = {Index{1}};
    let Inst{20} = {Index{0}};
    let Inst{19-16} = Re{3-0};
  }

  def _8h8h : NI_2VE<0b1, u, 0b01, opcode, asmop, "8h", "8h", "h",
                     neon_uimm3_bare, VPR128, VPR128, VPR128Lo> {
    let Inst{11} = {Index{2}};
    let Inst{21} = {Index{1}};
    let Inst{20} = {Index{0}};
    let Inst{19-16} = Re{3-0};
  }
}

defm MLAvve : NI_2VE_v1<0b1, 0b0000, "mla">;
defm MLSvve : NI_2VE_v1<0b1, 0b0100, "mls">;

// Pattern for lane in 128-bit vector
class NI_2VE_laneq<Instruction INST, Operand OpImm, SDPatternOperator op,
                   RegisterOperand ResVPR, RegisterOperand OpVPR,
                   RegisterOperand EleOpVPR, ValueType ResTy, ValueType OpTy,
                   ValueType EleOpTy, SDPatternOperator coreop>
  : Pat<(ResTy (op (ResTy ResVPR:$src), (OpTy OpVPR:$Rn),
          (OpTy (coreop (EleOpTy EleOpVPR:$Re), (i64 OpImm:$Index))))),
        (INST ResVPR:$src, OpVPR:$Rn, EleOpVPR:$Re, OpImm:$Index)>;

// Pattern for lane in 64-bit vector
class NI_2VE_lane<Instruction INST, Operand OpImm, SDPatternOperator op,
                  RegisterOperand ResVPR, RegisterOperand OpVPR,
                  RegisterOperand EleOpVPR, ValueType ResTy, ValueType OpTy,
                  ValueType EleOpTy, SDPatternOperator coreop>
  : Pat<(ResTy (op (ResTy ResVPR:$src), (OpTy OpVPR:$Rn),
          (OpTy (coreop (EleOpTy EleOpVPR:$Re), (i64 OpImm:$Index))))),
        (INST ResVPR:$src, OpVPR:$Rn, 
          (SUBREG_TO_REG (i64 0), EleOpVPR:$Re, sub_64), OpImm:$Index)>;

multiclass NI_2VE_v1_pat<string subop, SDPatternOperator op>
{
  def : NI_2VE_laneq<!cast<Instruction>(subop # "_2s4s"), neon_uimm2_bare,
                     op, VPR64, VPR64, VPR128, v2i32, v2i32, v4i32,
                     BinOpFrag<(Neon_vduplane
                                 (Neon_low4S node:$LHS), node:$RHS)>>;

  def : NI_2VE_laneq<!cast<Instruction>(subop # "_4s4s"), neon_uimm2_bare,
                     op, VPR128, VPR128, VPR128, v4i32, v4i32, v4i32,
                     BinOpFrag<(Neon_vduplane node:$LHS, node:$RHS)>>;

  def : NI_2VE_laneq<!cast<Instruction>(subop # "_4h8h"), neon_uimm3_bare,
                     op, VPR64, VPR64, VPR128Lo, v4i16, v4i16, v8i16,
                     BinOpFrag<(Neon_vduplane
                                 (Neon_low8H node:$LHS), node:$RHS)>>;

  def : NI_2VE_laneq<!cast<Instruction>(subop # "_8h8h"), neon_uimm3_bare,
                     op, VPR128, VPR128, VPR128Lo, v8i16, v8i16, v8i16,
                     BinOpFrag<(Neon_vduplane node:$LHS, node:$RHS)>>;

  // Index can only be half of the max value for lane in 64-bit vector

  def : NI_2VE_lane<!cast<Instruction>(subop # "_2s4s"), neon_uimm1_bare,
                    op, VPR64, VPR64, VPR64, v2i32, v2i32, v2i32,
                    BinOpFrag<(Neon_vduplane node:$LHS, node:$RHS)>>;

  def : NI_2VE_lane<!cast<Instruction>(subop # "_4s4s"), neon_uimm1_bare,
                    op, VPR128, VPR128, VPR64, v4i32, v4i32, v2i32,
                    BinOpFrag<(Neon_vduplane
                                (Neon_combine_4S node:$LHS, undef),
                                 node:$RHS)>>;

  def : NI_2VE_lane<!cast<Instruction>(subop # "_4h8h"), neon_uimm2_bare,
                    op, VPR64, VPR64, VPR64Lo, v4i16, v4i16, v4i16,
                    BinOpFrag<(Neon_vduplane node:$LHS, node:$RHS)>>;

  def : NI_2VE_lane<!cast<Instruction>(subop # "_8h8h"), neon_uimm2_bare,
                    op, VPR128, VPR128, VPR64Lo, v8i16, v8i16, v4i16,
                    BinOpFrag<(Neon_vduplane
                                (Neon_combine_8H node:$LHS, undef),
                                node:$RHS)>>;
}

defm MLA_lane_v1 : NI_2VE_v1_pat<"MLAvve", Neon_mla>;
defm MLS_lane_v1 : NI_2VE_v1_pat<"MLSvve", Neon_mls>;

class NI_2VE_2op<bit q, bit u, bits<2> size, bits<4> opcode,
                 string asmop, string ResS, string OpS, string EleOpS,
                 Operand OpImm, RegisterOperand ResVPR,
                 RegisterOperand OpVPR, RegisterOperand EleOpVPR>
  : NeonI_2VElem<q, u, size, opcode, 
                 (outs ResVPR:$Rd), (ins OpVPR:$Rn,
                                         EleOpVPR:$Re, OpImm:$Index),
                 asmop # "\t$Rd." # ResS # ", $Rn." # OpS #
                 ", $Re." # EleOpS # "[$Index]",
                 [],
                 NoItinerary> {
  bits<3> Index;
  bits<5> Re;
}

multiclass NI_2VE_v1_2op<bit u, bits<4> opcode, string asmop>
{
  // vector register class for element is always 128-bit to cover the max index
  def _2s4s : NI_2VE_2op<0b0, u, 0b10, opcode, asmop, "2s", "2s", "s",
                         neon_uimm2_bare, VPR64, VPR64, VPR128> {
    let Inst{11} = {Index{1}};
    let Inst{21} = {Index{0}};
    let Inst{20-16} = Re;
  }

  def _4s4s : NI_2VE_2op<0b1, u, 0b10, opcode, asmop, "4s", "4s", "s",
                         neon_uimm2_bare, VPR128, VPR128, VPR128> {
    let Inst{11} = {Index{1}};
    let Inst{21} = {Index{0}};
    let Inst{20-16} = Re;
  }

  // Index operations on 16-bit(H) elements are restricted to using v0-v15.
  def _4h8h : NI_2VE_2op<0b0, u, 0b01, opcode, asmop, "4h", "4h", "h",
                         neon_uimm3_bare, VPR64, VPR64, VPR128Lo> {
    let Inst{11} = {Index{2}};
    let Inst{21} = {Index{1}};
    let Inst{20} = {Index{0}};
    let Inst{19-16} = Re{3-0};
  }

  def _8h8h : NI_2VE_2op<0b1, u, 0b01, opcode, asmop, "8h", "8h", "h",
                         neon_uimm3_bare, VPR128, VPR128, VPR128Lo> {
    let Inst{11} = {Index{2}};
    let Inst{21} = {Index{1}};
    let Inst{20} = {Index{0}};
    let Inst{19-16} = Re{3-0};
  }
}

defm MULve : NI_2VE_v1_2op<0b0, 0b1000, "mul">;
defm SQDMULHve : NI_2VE_v1_2op<0b0, 0b1100, "sqdmulh">;
defm SQRDMULHve : NI_2VE_v1_2op<0b0, 0b1101, "sqrdmulh">;

// Pattern for lane in 128-bit vector
class NI_2VE_mul_laneq<Instruction INST, Operand OpImm, SDPatternOperator op,
                       RegisterOperand OpVPR, RegisterOperand EleOpVPR,
                       ValueType ResTy, ValueType OpTy, ValueType EleOpTy,
                       SDPatternOperator coreop>
  : Pat<(ResTy (op (OpTy OpVPR:$Rn),
          (OpTy (coreop (EleOpTy EleOpVPR:$Re), (i64 OpImm:$Index))))),
        (INST OpVPR:$Rn, EleOpVPR:$Re, OpImm:$Index)>;

// Pattern for lane in 64-bit vector
class NI_2VE_mul_lane<Instruction INST, Operand OpImm, SDPatternOperator op,
                      RegisterOperand OpVPR, RegisterOperand EleOpVPR,
                      ValueType ResTy, ValueType OpTy, ValueType EleOpTy,
                      SDPatternOperator coreop>
  : Pat<(ResTy (op (OpTy OpVPR:$Rn),
          (OpTy (coreop (EleOpTy EleOpVPR:$Re), (i64 OpImm:$Index))))),
        (INST OpVPR:$Rn, 
          (SUBREG_TO_REG (i64 0), EleOpVPR:$Re, sub_64), OpImm:$Index)>;

multiclass NI_2VE_mul_v1_pat<string subop, SDPatternOperator op>
{
  def : NI_2VE_mul_laneq<!cast<Instruction>(subop # "_2s4s"), neon_uimm2_bare,
                         op, VPR64, VPR128, v2i32, v2i32, v4i32,
                         BinOpFrag<(Neon_vduplane
                                     (Neon_low4S node:$LHS), node:$RHS)>>;

  def : NI_2VE_mul_laneq<!cast<Instruction>(subop # "_4s4s"), neon_uimm2_bare,
                         op, VPR128, VPR128, v4i32, v4i32, v4i32,
                         BinOpFrag<(Neon_vduplane node:$LHS, node:$RHS)>>;

  def : NI_2VE_mul_laneq<!cast<Instruction>(subop # "_4h8h"), neon_uimm3_bare,
                         op, VPR64, VPR128Lo, v4i16, v4i16, v8i16,
                         BinOpFrag<(Neon_vduplane
                                    (Neon_low8H node:$LHS), node:$RHS)>>;

  def : NI_2VE_mul_laneq<!cast<Instruction>(subop # "_8h8h"), neon_uimm3_bare,
                         op, VPR128, VPR128Lo, v8i16, v8i16, v8i16,
                         BinOpFrag<(Neon_vduplane node:$LHS, node:$RHS)>>;

  // Index can only be half of the max value for lane in 64-bit vector

  def : NI_2VE_mul_lane<!cast<Instruction>(subop # "_2s4s"), neon_uimm1_bare,
                        op, VPR64, VPR64, v2i32, v2i32, v2i32,
                        BinOpFrag<(Neon_vduplane node:$LHS, node:$RHS)>>;

  def : NI_2VE_mul_lane<!cast<Instruction>(subop # "_4s4s"), neon_uimm1_bare,
                        op, VPR128, VPR64, v4i32, v4i32, v2i32,
                        BinOpFrag<(Neon_vduplane
                                    (Neon_combine_4S node:$LHS, undef),
                                     node:$RHS)>>;

  def : NI_2VE_mul_lane<!cast<Instruction>(subop # "_4h8h"), neon_uimm2_bare,
                        op, VPR64, VPR64Lo, v4i16, v4i16, v4i16,
                        BinOpFrag<(Neon_vduplane node:$LHS, node:$RHS)>>;

  def : NI_2VE_mul_lane<!cast<Instruction>(subop # "_8h8h"), neon_uimm2_bare,
                        op, VPR128, VPR64Lo, v8i16, v8i16, v4i16,
                        BinOpFrag<(Neon_vduplane
                                    (Neon_combine_8H node:$LHS, undef),
                                    node:$RHS)>>;
}

defm MUL_lane_v1 : NI_2VE_mul_v1_pat<"MULve", mul>;
defm SQDMULH_lane_v1 : NI_2VE_mul_v1_pat<"SQDMULHve", int_arm_neon_vqdmulh>;
defm SQRDMULH_lane_v1 : NI_2VE_mul_v1_pat<"SQRDMULHve", int_arm_neon_vqrdmulh>;

// Variant 2

multiclass NI_2VE_v2_2op<bit u, bits<4> opcode, string asmop>
{
  // vector register class for element is always 128-bit to cover the max index
  def _2s4s : NI_2VE_2op<0b0, u, 0b10, opcode, asmop, "2s", "2s", "s",
                         neon_uimm2_bare, VPR64, VPR64, VPR128> {
    let Inst{11} = {Index{1}};
    let Inst{21} = {Index{0}};
    let Inst{20-16} = Re;
  }

  def _4s4s : NI_2VE_2op<0b1, u, 0b10, opcode, asmop, "4s", "4s", "s",
                         neon_uimm2_bare, VPR128, VPR128, VPR128> {
    let Inst{11} = {Index{1}};
    let Inst{21} = {Index{0}};
    let Inst{20-16} = Re;
  }

  // _1d2d doesn't exist!

  def _2d2d : NI_2VE_2op<0b1, u, 0b11, opcode, asmop, "2d", "2d", "d",
                         neon_uimm1_bare, VPR128, VPR128, VPR128> {
    let Inst{11} = {Index{0}};
    let Inst{21} = 0b0;
    let Inst{20-16} = Re;
  }
}

defm FMULve : NI_2VE_v2_2op<0b0, 0b1001, "fmul">;
defm FMULXve : NI_2VE_v2_2op<0b1, 0b1001, "fmulx">;

class NI_2VE_mul_lane_2d<Instruction INST, Operand OpImm, SDPatternOperator op,
                         RegisterOperand OpVPR, RegisterOperand EleOpVPR,
                         ValueType ResTy, ValueType OpTy, ValueType EleOpTy,
                         SDPatternOperator coreop>
  : Pat<(ResTy (op (OpTy OpVPR:$Rn),
          (OpTy (coreop (EleOpTy EleOpVPR:$Re), (EleOpTy EleOpVPR:$Re))))),
        (INST OpVPR:$Rn, 
          (SUBREG_TO_REG (i64 0), EleOpVPR:$Re, sub_64), 0)>;

multiclass NI_2VE_mul_v2_pat<string subop, SDPatternOperator op>
{
  def : NI_2VE_mul_laneq<!cast<Instruction>(subop # "_2s4s"), neon_uimm2_bare,
                         op, VPR64, VPR128, v2f32, v2f32, v4f32,
                         BinOpFrag<(Neon_vduplane
                                     (Neon_low4f node:$LHS), node:$RHS)>>;

  def : NI_2VE_mul_laneq<!cast<Instruction>(subop # "_4s4s"), neon_uimm2_bare,
                         op, VPR128, VPR128, v4f32, v4f32, v4f32,
                         BinOpFrag<(Neon_vduplane node:$LHS, node:$RHS)>>;

  def : NI_2VE_mul_laneq<!cast<Instruction>(subop # "_2d2d"), neon_uimm1_bare,
                         op, VPR128, VPR128, v2f64, v2f64, v2f64,
                         BinOpFrag<(Neon_vduplane node:$LHS, node:$RHS)>>;

  // Index can only be half of the max value for lane in 64-bit vector

  def : NI_2VE_mul_lane<!cast<Instruction>(subop # "_2s4s"), neon_uimm1_bare,
                        op, VPR64, VPR64, v2f32, v2f32, v2f32,
                        BinOpFrag<(Neon_vduplane node:$LHS, node:$RHS)>>;

  def : NI_2VE_mul_lane<!cast<Instruction>(subop # "_4s4s"), neon_uimm1_bare,
                        op, VPR128, VPR64, v4f32, v4f32, v2f32,
                        BinOpFrag<(Neon_vduplane
                                    (Neon_combine_4f node:$LHS, undef),
                                    node:$RHS)>>;

  def : NI_2VE_mul_lane_2d<!cast<Instruction>(subop # "_2d2d"), neon_uimm1_bare,
                           op, VPR128, VPR64, v2f64, v2f64, v1f64,
                           BinOpFrag<(Neon_combine_2d node:$LHS, node:$RHS)>>;
}

defm FMUL_lane_v2 : NI_2VE_mul_v2_pat<"FMULve", fmul>;
defm FMULX_lane_v2 : NI_2VE_mul_v2_pat<"FMULXve", int_aarch64_neon_vmulx>;

// The followings are patterns using fma
// -ffp-contract=fast generates fma

multiclass NI_2VE_v2<bit u, bits<4> opcode, string asmop>
{
  // vector register class for element is always 128-bit to cover the max index
  def _2s4s : NI_2VE<0b0, u, 0b10, opcode, asmop, "2s", "2s", "s",
                     neon_uimm2_bare, VPR64, VPR64, VPR128> {
    let Inst{11} = {Index{1}};
    let Inst{21} = {Index{0}};
    let Inst{20-16} = Re;
  }

  def _4s4s : NI_2VE<0b1, u, 0b10, opcode, asmop, "4s", "4s", "s",
                     neon_uimm2_bare, VPR128, VPR128, VPR128> {
    let Inst{11} = {Index{1}};
    let Inst{21} = {Index{0}};
    let Inst{20-16} = Re;
  }

  // _1d2d doesn't exist!
  
  def _2d2d : NI_2VE<0b1, u, 0b11, opcode, asmop, "2d", "2d", "d",
                     neon_uimm1_bare, VPR128, VPR128, VPR128> {
    let Inst{11} = {Index{0}};
    let Inst{21} = 0b0;
    let Inst{20-16} = Re;
  }
}

defm FMLAvve : NI_2VE_v2<0b0, 0b0001, "fmla">;
defm FMLSvve : NI_2VE_v2<0b0, 0b0101, "fmls">;

// Pattern for lane in 128-bit vector
class NI_2VEswap_laneq<Instruction INST, Operand OpImm, SDPatternOperator op,
                       RegisterOperand ResVPR, RegisterOperand OpVPR,
                       ValueType ResTy, ValueType OpTy,
                       SDPatternOperator coreop>
  : Pat<(ResTy (op (ResTy (coreop (OpTy OpVPR:$Re), (i64 OpImm:$Index))),
                   (ResTy ResVPR:$src), (ResTy ResVPR:$Rn))),
        (INST ResVPR:$src, ResVPR:$Rn, OpVPR:$Re, OpImm:$Index)>;

// Pattern for lane in 64-bit vector
class NI_2VEswap_lane<Instruction INST, Operand OpImm, SDPatternOperator op,
                      RegisterOperand ResVPR, RegisterOperand OpVPR,
                      ValueType ResTy, ValueType OpTy,
                      SDPatternOperator coreop>
  : Pat<(ResTy (op (ResTy (coreop (OpTy OpVPR:$Re), (i64 OpImm:$Index))),
                   (ResTy ResVPR:$Rn), (ResTy ResVPR:$src))),
        (INST ResVPR:$src, ResVPR:$Rn, 
          (SUBREG_TO_REG (i64 0), OpVPR:$Re, sub_64), OpImm:$Index)>;

// Pattern for lane in 64-bit vector
class NI_2VEswap_lane_2d2d<Instruction INST, Operand OpImm,
                           SDPatternOperator op,
                           RegisterOperand ResVPR, RegisterOperand OpVPR,
                           ValueType ResTy, ValueType OpTy,
                           SDPatternOperator coreop>
  : Pat<(ResTy (op (ResTy (coreop (OpTy OpVPR:$Re), (OpTy OpVPR:$Re))),
                   (ResTy ResVPR:$Rn), (ResTy ResVPR:$src))),
        (INST ResVPR:$src, ResVPR:$Rn, 
          (SUBREG_TO_REG (i64 0), OpVPR:$Re, sub_64), 0)>;


multiclass NI_2VE_fma_v2_pat<string subop, SDPatternOperator op>
{
  def : NI_2VEswap_laneq<!cast<Instruction>(subop # "_2s4s"),
                         neon_uimm2_bare, op, VPR64, VPR128, v2f32, v4f32,
                         BinOpFrag<(Neon_vduplane
                                     (Neon_low4f node:$LHS), node:$RHS)>>;

  def : NI_2VEswap_laneq<!cast<Instruction>(subop # "_4s4s"),
                         neon_uimm2_bare, op, VPR128, VPR128, v4f32, v4f32,
                         BinOpFrag<(Neon_vduplane node:$LHS, node:$RHS)>>;

  def : NI_2VEswap_laneq<!cast<Instruction>(subop # "_2d2d"),
                         neon_uimm1_bare, op, VPR128, VPR128, v2f64, v2f64,
                         BinOpFrag<(Neon_vduplane node:$LHS, node:$RHS)>>;

  // Index can only be half of the max value for lane in 64-bit vector

  def : NI_2VEswap_lane<!cast<Instruction>(subop # "_2s4s"),
                        neon_uimm1_bare, op, VPR64, VPR64, v2f32, v2f32,
                        BinOpFrag<(Neon_vduplane node:$LHS, node:$RHS)>>;

  def : NI_2VEswap_lane<!cast<Instruction>(subop # "_4s4s"),
                        neon_uimm1_bare, op, VPR128, VPR64, v4f32, v2f32,
                        BinOpFrag<(Neon_vduplane
                                    (Neon_combine_4f node:$LHS, undef),
                                    node:$RHS)>>;

  def : NI_2VEswap_lane_2d2d<!cast<Instruction>(subop # "_2d2d"),
                             neon_uimm1_bare, op, VPR128, VPR64, v2f64, v1f64,
                             BinOpFrag<(Neon_combine_2d node:$LHS, node:$RHS)>>;
}

defm FMLA_lane_v2_s : NI_2VE_fma_v2_pat<"FMLAvve", fma>;

multiclass NI_2VE_fms_v2_pat<string subop, SDPatternOperator op>
{
  def : NI_2VEswap_laneq<!cast<Instruction>(subop # "_2s4s"),
                         neon_uimm2_bare, op, VPR64, VPR128, v2f32, v4f32,
                         BinOpFrag<(fneg (Neon_vduplane
                                     (Neon_low4f node:$LHS), node:$RHS))>>;

  def : NI_2VEswap_laneq<!cast<Instruction>(subop # "_2s4s"),
                         neon_uimm2_bare, op, VPR64, VPR128, v2f32, v4f32,
                         BinOpFrag<(Neon_vduplane
                                     (Neon_low4f (fneg node:$LHS)),
                                     node:$RHS)>>;

  def : NI_2VEswap_laneq<!cast<Instruction>(subop # "_4s4s"),
                         neon_uimm2_bare, op, VPR128, VPR128, v4f32, v4f32,
                         BinOpFrag<(fneg (Neon_vduplane
                                     node:$LHS, node:$RHS))>>;

  def : NI_2VEswap_laneq<!cast<Instruction>(subop # "_4s4s"),
                         neon_uimm2_bare, op, VPR128, VPR128, v4f32, v4f32,
                         BinOpFrag<(Neon_vduplane
                                     (fneg node:$LHS), node:$RHS)>>;

  def : NI_2VEswap_laneq<!cast<Instruction>(subop # "_2d2d"),
                         neon_uimm1_bare, op, VPR128, VPR128, v2f64, v2f64,
                         BinOpFrag<(fneg (Neon_vduplane
                                     node:$LHS, node:$RHS))>>;

  def : NI_2VEswap_laneq<!cast<Instruction>(subop # "_2d2d"),
                         neon_uimm1_bare, op, VPR128, VPR128, v2f64, v2f64,
                         BinOpFrag<(Neon_vduplane
                                     (fneg node:$LHS), node:$RHS)>>;

  // Index can only be half of the max value for lane in 64-bit vector

  def : NI_2VEswap_lane<!cast<Instruction>(subop # "_2s4s"),
                        neon_uimm1_bare, op, VPR64, VPR64, v2f32, v2f32,
                        BinOpFrag<(fneg (Neon_vduplane
                                    node:$LHS, node:$RHS))>>;

  def : NI_2VEswap_lane<!cast<Instruction>(subop # "_2s4s"),
                        neon_uimm1_bare, op, VPR64, VPR64, v2f32, v2f32,
                        BinOpFrag<(Neon_vduplane
                                    (fneg node:$LHS), node:$RHS)>>;

  def : NI_2VEswap_lane<!cast<Instruction>(subop # "_4s4s"),
                        neon_uimm1_bare, op, VPR128, VPR64, v4f32, v2f32,
                        BinOpFrag<(fneg (Neon_vduplane
                                    (Neon_combine_4f node:$LHS, undef),
                                    node:$RHS))>>;

  def : NI_2VEswap_lane<!cast<Instruction>(subop # "_4s4s"),
                        neon_uimm1_bare, op, VPR128, VPR64, v4f32, v2f32,
                        BinOpFrag<(Neon_vduplane
                                    (Neon_combine_4f (fneg node:$LHS), undef),
                                    node:$RHS)>>;

  def : NI_2VEswap_lane_2d2d<!cast<Instruction>(subop # "_2d2d"),
                             neon_uimm1_bare, op, VPR128, VPR64, v2f64, v1f64,
                             BinOpFrag<(fneg (Neon_combine_2d
                                         node:$LHS, node:$RHS))>>;

  def : NI_2VEswap_lane_2d2d<!cast<Instruction>(subop # "_2d2d"),
                             neon_uimm1_bare, op, VPR128, VPR64, v2f64, v1f64,
                             BinOpFrag<(Neon_combine_2d
                                         (fneg node:$LHS), (fneg node:$RHS))>>;
}

defm FMLS_lane_v2_s : NI_2VE_fms_v2_pat<"FMLSvve", fma>;

// Variant 3: Long type
// E.g. SMLAL : 4S/4H/H (v0-v15), 2D/2S/S
//      SMLAL2: 4S/8H/H (v0-v15), 2D/4S/S

multiclass NI_2VE_v3<bit u, bits<4> opcode, string asmop>
{
  // vector register class for element is always 128-bit to cover the max index
  def _2d2s : NI_2VE<0b0, u, 0b10, opcode, asmop, "2d", "2s", "s",
                     neon_uimm2_bare, VPR128, VPR64, VPR128> {
    let Inst{11} = {Index{1}};
    let Inst{21} = {Index{0}};
    let Inst{20-16} = Re;
  }
  
  def _2d4s : NI_2VE<0b1, u, 0b10, opcode, asmop # "2", "2d", "4s", "s",
                     neon_uimm2_bare, VPR128, VPR128, VPR128> {
    let Inst{11} = {Index{1}};
    let Inst{21} = {Index{0}};
    let Inst{20-16} = Re;
  }

  // Index operations on 16-bit(H) elements are restricted to using v0-v15.
  def _4s8h : NI_2VE<0b1, u, 0b01, opcode, asmop # "2", "4s", "8h", "h",
                     neon_uimm3_bare, VPR128, VPR128, VPR128Lo> {
    let Inst{11} = {Index{2}};
    let Inst{21} = {Index{1}};
    let Inst{20} = {Index{0}};
    let Inst{19-16} = Re{3-0};
  }
  
  def _4s4h : NI_2VE<0b0, u, 0b01, opcode, asmop, "4s", "4h", "h",
                     neon_uimm3_bare, VPR128, VPR64, VPR128Lo> {
    let Inst{11} = {Index{2}};
    let Inst{21} = {Index{1}};
    let Inst{20} = {Index{0}};
    let Inst{19-16} = Re{3-0};
  }
}

defm SMLALvve : NI_2VE_v3<0b0, 0b0010, "smlal">;
defm UMLALvve : NI_2VE_v3<0b1, 0b0010, "umlal">;
defm SMLSLvve : NI_2VE_v3<0b0, 0b0110, "smlsl">;
defm UMLSLvve : NI_2VE_v3<0b1, 0b0110, "umlsl">;
defm SQDMLALvve : NI_2VE_v3<0b0, 0b0011, "sqdmlal">;
defm SQDMLSLvve : NI_2VE_v3<0b0, 0b0111, "sqdmlsl">;

multiclass NI_2VE_v3_2op<bit u, bits<4> opcode, string asmop>
{
  // vector register class for element is always 128-bit to cover the max index
  def _2d2s : NI_2VE_2op<0b0, u, 0b10, opcode, asmop, "2d", "2s", "s",
                         neon_uimm2_bare, VPR128, VPR64, VPR128> {
    let Inst{11} = {Index{1}};
    let Inst{21} = {Index{0}};
    let Inst{20-16} = Re;
  }
  
  def _2d4s : NI_2VE_2op<0b1, u, 0b10, opcode, asmop # "2", "2d", "4s", "s",
                         neon_uimm2_bare, VPR128, VPR128, VPR128> {
    let Inst{11} = {Index{1}};
    let Inst{21} = {Index{0}};
    let Inst{20-16} = Re;
  }

  // Index operations on 16-bit(H) elements are restricted to using v0-v15.
  def _4s8h : NI_2VE_2op<0b1, u, 0b01, opcode, asmop # "2", "4s", "8h", "h",
                         neon_uimm3_bare, VPR128, VPR128, VPR128Lo> {
    let Inst{11} = {Index{2}};
    let Inst{21} = {Index{1}};
    let Inst{20} = {Index{0}};
    let Inst{19-16} = Re{3-0};
  }
  
  def _4s4h : NI_2VE_2op<0b0, u, 0b01, opcode, asmop, "4s", "4h", "h",
                         neon_uimm3_bare, VPR128, VPR64, VPR128Lo> {
    let Inst{11} = {Index{2}};
    let Inst{21} = {Index{1}};
    let Inst{20} = {Index{0}};
    let Inst{19-16} = Re{3-0};
  }
}

defm SMULLve : NI_2VE_v3_2op<0b0, 0b1010, "smull">;
defm UMULLve : NI_2VE_v3_2op<0b1, 0b1010, "umull">;
defm SQDMULLve : NI_2VE_v3_2op<0b0, 0b1011, "sqdmull">;

// Pattern for lane in 128-bit vector
class NI_2VEL2_laneq<Instruction INST, Operand OpImm, SDPatternOperator op,
                     RegisterOperand EleOpVPR, ValueType ResTy,
                     ValueType OpTy, ValueType EleOpTy, ValueType HalfOpTy,
                     SDPatternOperator hiop, SDPatternOperator coreop>
  : Pat<(ResTy (op (ResTy VPR128:$src),
          (HalfOpTy (hiop (OpTy VPR128:$Rn))),
          (HalfOpTy (coreop (EleOpTy EleOpVPR:$Re), (i64 OpImm:$Index))))),
        (INST VPR128:$src, VPR128:$Rn, EleOpVPR:$Re, OpImm:$Index)>;

// Pattern for lane in 64-bit vector
class NI_2VEL2_lane<Instruction INST, Operand OpImm, SDPatternOperator op,
                    RegisterOperand EleOpVPR, ValueType ResTy,
                    ValueType OpTy, ValueType EleOpTy, ValueType HalfOpTy,
                    SDPatternOperator hiop, SDPatternOperator coreop>
  : Pat<(ResTy (op (ResTy VPR128:$src),
          (HalfOpTy (hiop (OpTy VPR128:$Rn))),
          (HalfOpTy (coreop (EleOpTy EleOpVPR:$Re), (i64 OpImm:$Index))))),
        (INST VPR128:$src, VPR128:$Rn, 
          (SUBREG_TO_REG (i64 0), EleOpVPR:$Re, sub_64), OpImm:$Index)>;

multiclass NI_2VEL_v3_pat<string subop, SDPatternOperator op>
{
  def : NI_2VE_laneq<!cast<Instruction>(subop # "_4s4h"), neon_uimm3_bare,
                     op, VPR128, VPR64, VPR128Lo, v4i32, v4i16, v8i16,
                     BinOpFrag<(Neon_vduplane
                                 (Neon_low8H node:$LHS), node:$RHS)>>;
  
  def : NI_2VE_laneq<!cast<Instruction>(subop # "_2d2s"), neon_uimm2_bare,
                     op, VPR128, VPR64, VPR128, v2i64, v2i32, v4i32,
                     BinOpFrag<(Neon_vduplane
                                 (Neon_low4S node:$LHS), node:$RHS)>>;
  
  def : NI_2VEL2_laneq<!cast<Instruction>(subop # "_4s8h"), neon_uimm3_bare,
                       op, VPR128Lo, v4i32, v8i16, v8i16, v4i16, Neon_High8H,
                       BinOpFrag<(Neon_vduplane
                                   (Neon_low8H node:$LHS), node:$RHS)>>;
  
  def : NI_2VEL2_laneq<!cast<Instruction>(subop # "_2d4s"), neon_uimm2_bare,
                       op, VPR128, v2i64, v4i32, v4i32, v2i32, Neon_High4S,
                       BinOpFrag<(Neon_vduplane
                                   (Neon_low4S node:$LHS), node:$RHS)>>;
  
  // Index can only be half of the max value for lane in 64-bit vector

  def : NI_2VE_lane<!cast<Instruction>(subop # "_4s4h"), neon_uimm2_bare,
                    op, VPR128, VPR64, VPR64Lo, v4i32, v4i16, v4i16,
                    BinOpFrag<(Neon_vduplane node:$LHS, node:$RHS)>>;
  
  def : NI_2VE_lane<!cast<Instruction>(subop # "_2d2s"), neon_uimm1_bare,
                    op, VPR128, VPR64, VPR64, v2i64, v2i32, v2i32,
                    BinOpFrag<(Neon_vduplane node:$LHS, node:$RHS)>>;

  def : NI_2VEL2_lane<!cast<Instruction>(subop # "_4s8h"), neon_uimm2_bare,
                      op, VPR64Lo, v4i32, v8i16, v4i16, v4i16, Neon_High8H,
                      BinOpFrag<(Neon_vduplane node:$LHS, node:$RHS)>>;
  
  def : NI_2VEL2_lane<!cast<Instruction>(subop # "_2d4s"), neon_uimm1_bare,
                      op, VPR64, v2i64, v4i32, v2i32, v2i32, Neon_High4S,
                      BinOpFrag<(Neon_vduplane node:$LHS, node:$RHS)>>;
}

defm SMLAL_lane_v3 : NI_2VEL_v3_pat<"SMLALvve", Neon_smlal>;
defm UMLAL_lane_v3 : NI_2VEL_v3_pat<"UMLALvve", Neon_umlal>;
defm SMLSL_lane_v3 : NI_2VEL_v3_pat<"SMLSLvve", Neon_smlsl>;
defm UMLSL_lane_v3 : NI_2VEL_v3_pat<"UMLSLvve", Neon_umlsl>;

// Pattern for lane in 128-bit vector
class NI_2VEL2_mul_laneq<Instruction INST, Operand OpImm, SDPatternOperator op,
                         RegisterOperand EleOpVPR, ValueType ResTy,
                         ValueType OpTy, ValueType EleOpTy, ValueType HalfOpTy,
                         SDPatternOperator hiop, SDPatternOperator coreop>
  : Pat<(ResTy (op 
          (HalfOpTy (hiop (OpTy VPR128:$Rn))),
          (HalfOpTy (coreop (EleOpTy EleOpVPR:$Re), (i64 OpImm:$Index))))),
        (INST VPR128:$Rn, EleOpVPR:$Re, OpImm:$Index)>;

// Pattern for lane in 64-bit vector
class NI_2VEL2_mul_lane<Instruction INST, Operand OpImm, SDPatternOperator op,
                        RegisterOperand EleOpVPR, ValueType ResTy,
                        ValueType OpTy, ValueType EleOpTy, ValueType HalfOpTy,
                        SDPatternOperator hiop, SDPatternOperator coreop>
  : Pat<(ResTy (op
          (HalfOpTy (hiop (OpTy VPR128:$Rn))),
          (HalfOpTy (coreop (EleOpTy EleOpVPR:$Re), (i64 OpImm:$Index))))),
        (INST VPR128:$Rn, 
          (SUBREG_TO_REG (i64 0), EleOpVPR:$Re, sub_64), OpImm:$Index)>;

multiclass NI_2VEL_mul_v3_pat<string subop, SDPatternOperator op>
{
  def : NI_2VE_mul_laneq<!cast<Instruction>(subop # "_4s4h"), neon_uimm3_bare,
                         op, VPR64, VPR128Lo, v4i32, v4i16, v8i16,
                         BinOpFrag<(Neon_vduplane
                                     (Neon_low8H node:$LHS), node:$RHS)>>;

  def : NI_2VE_mul_laneq<!cast<Instruction>(subop # "_2d2s"), neon_uimm2_bare,
                         op, VPR64, VPR128, v2i64, v2i32, v4i32,
                         BinOpFrag<(Neon_vduplane
                                     (Neon_low4S node:$LHS), node:$RHS)>>;

  def : NI_2VEL2_mul_laneq<!cast<Instruction>(subop # "_4s8h"), neon_uimm3_bare,
                           op, VPR128Lo, v4i32, v8i16, v8i16, v4i16,
                           Neon_High8H,
                           BinOpFrag<(Neon_vduplane
                                       (Neon_low8H node:$LHS), node:$RHS)>>;
  
  def : NI_2VEL2_mul_laneq<!cast<Instruction>(subop # "_2d4s"), neon_uimm2_bare,
                           op, VPR128, v2i64, v4i32, v4i32, v2i32, Neon_High4S,
                           BinOpFrag<(Neon_vduplane
                                       (Neon_low4S node:$LHS), node:$RHS)>>;
  
  // Index can only be half of the max value for lane in 64-bit vector

  def : NI_2VE_mul_lane<!cast<Instruction>(subop # "_4s4h"), neon_uimm2_bare,
                        op, VPR64, VPR64Lo, v4i32, v4i16, v4i16,
                        BinOpFrag<(Neon_vduplane node:$LHS, node:$RHS)>>;

  def : NI_2VE_mul_lane<!cast<Instruction>(subop # "_2d2s"), neon_uimm1_bare,
                        op, VPR64, VPR64, v2i64, v2i32, v2i32,
                        BinOpFrag<(Neon_vduplane node:$LHS, node:$RHS)>>;

  def : NI_2VEL2_mul_lane<!cast<Instruction>(subop # "_4s8h"), neon_uimm2_bare,
                          op, VPR64Lo, v4i32, v8i16, v4i16, v4i16, Neon_High8H,
                          BinOpFrag<(Neon_vduplane node:$LHS, node:$RHS)>>;
  
  def : NI_2VEL2_mul_lane<!cast<Instruction>(subop # "_2d4s"), neon_uimm1_bare,
                          op, VPR64, v2i64, v4i32, v2i32, v2i32, Neon_High4S,
                          BinOpFrag<(Neon_vduplane node:$LHS, node:$RHS)>>;
}

defm SMULL_lane_v3 : NI_2VEL_mul_v3_pat<"SMULLve", int_arm_neon_vmulls>;
defm UMULL_lane_v3 : NI_2VEL_mul_v3_pat<"UMULLve", int_arm_neon_vmullu>;
defm SQDMULL_lane_v3 : NI_2VEL_mul_v3_pat<"SQDMULLve", int_arm_neon_vqdmull>;

multiclass NI_qdma<SDPatternOperator op>
{
  def _4s : PatFrag<(ops node:$Ra, node:$Rn, node:$Rm),
                    (op node:$Ra,
                      (v4i32 (int_arm_neon_vqdmull node:$Rn, node:$Rm)))>;

  def _2d : PatFrag<(ops node:$Ra, node:$Rn, node:$Rm),
                    (op node:$Ra,
                      (v2i64 (int_arm_neon_vqdmull node:$Rn, node:$Rm)))>;
}

defm Neon_qdmlal : NI_qdma<int_arm_neon_vqadds>;
defm Neon_qdmlsl : NI_qdma<int_arm_neon_vqsubs>;

multiclass NI_2VEL_v3_qdma_pat<string subop, string op>
{
  def : NI_2VE_laneq<!cast<Instruction>(subop # "_4s4h"), neon_uimm3_bare,
                     !cast<PatFrag>(op # "_4s"), VPR128, VPR64, VPR128Lo,
                     v4i32, v4i16, v8i16,
                     BinOpFrag<(Neon_vduplane
                                 (Neon_low8H node:$LHS), node:$RHS)>>;
  
  def : NI_2VE_laneq<!cast<Instruction>(subop # "_2d2s"), neon_uimm2_bare,
                     !cast<PatFrag>(op # "_2d"), VPR128, VPR64, VPR128,
                     v2i64, v2i32, v4i32,
                     BinOpFrag<(Neon_vduplane
                                 (Neon_low4S node:$LHS), node:$RHS)>>;
  
  def : NI_2VEL2_laneq<!cast<Instruction>(subop # "_4s8h"), neon_uimm3_bare,
                       !cast<PatFrag>(op # "_4s"), VPR128Lo,
                       v4i32, v8i16, v8i16, v4i16, Neon_High8H,
                       BinOpFrag<(Neon_vduplane
                                   (Neon_low8H node:$LHS), node:$RHS)>>;
  
  def : NI_2VEL2_laneq<!cast<Instruction>(subop # "_2d4s"), neon_uimm2_bare,
                       !cast<PatFrag>(op # "_2d"), VPR128,
                       v2i64, v4i32, v4i32, v2i32, Neon_High4S,
                       BinOpFrag<(Neon_vduplane
                                   (Neon_low4S node:$LHS), node:$RHS)>>;
  
  // Index can only be half of the max value for lane in 64-bit vector

  def : NI_2VE_lane<!cast<Instruction>(subop # "_4s4h"), neon_uimm2_bare,
                    !cast<PatFrag>(op # "_4s"), VPR128, VPR64, VPR64Lo,
                    v4i32, v4i16, v4i16,
                    BinOpFrag<(Neon_vduplane node:$LHS, node:$RHS)>>;
  
  def : NI_2VE_lane<!cast<Instruction>(subop # "_2d2s"), neon_uimm1_bare,
                    !cast<PatFrag>(op # "_2d"), VPR128, VPR64, VPR64,
                    v2i64, v2i32, v2i32,
                    BinOpFrag<(Neon_vduplane node:$LHS, node:$RHS)>>;

  def : NI_2VEL2_lane<!cast<Instruction>(subop # "_4s8h"), neon_uimm2_bare,
                      !cast<PatFrag>(op # "_4s"), VPR64Lo,
                      v4i32, v8i16, v4i16, v4i16, Neon_High8H,
                      BinOpFrag<(Neon_vduplane node:$LHS, node:$RHS)>>;
  
  def : NI_2VEL2_lane<!cast<Instruction>(subop # "_2d4s"), neon_uimm1_bare,
                      !cast<PatFrag>(op # "_2d"), VPR64,
                      v2i64, v4i32, v2i32, v2i32, Neon_High4S,
                      BinOpFrag<(Neon_vduplane node:$LHS, node:$RHS)>>;
}

defm SQDMLAL_lane_v3 : NI_2VEL_v3_qdma_pat<"SQDMLALvve", "Neon_qdmlal">;
defm SQDMLSL_lane_v3 : NI_2VEL_v3_qdma_pat<"SQDMLSLvve", "Neon_qdmlsl">;

// End of implementation for instruction class (3V Elem)

//Insert element (vector, from main)
def INSbw : NeonI_INS_main<"ins", "b", v16i8, GPR32, i32,
                           neon_uimm4_bare> {
  let Inst{20-16} = {Imm{3}, Imm{2}, Imm{1}, Imm{0}, 0b1};
}
def INShw : NeonI_INS_main<"ins", "h", v8i16, GPR32, i32,
                           neon_uimm3_bare> {
  let Inst{20-16} = {Imm{2}, Imm{1}, Imm{0}, 0b1, 0b0};
}
def INSsw : NeonI_INS_main<"ins", "s", v4i32, GPR32, i32,
                           neon_uimm2_bare> {
  let Inst{20-16} = {Imm{1}, Imm{0}, 0b1, 0b0, 0b0};
}
def INSdx : NeonI_INS_main<"ins", "d", v2i64, GPR64, i64,
                           neon_uimm1_bare> {
  let Inst{20-16} = {Imm, 0b1, 0b0, 0b0, 0b0};
}

class Neon_INS_main_pattern <ValueType ResTy,ValueType ExtResTy,
                             RegisterClass OpGPR, ValueType OpTy, 
                             Operand OpImm, Instruction INS> 
  : Pat<(ResTy (vector_insert
              (ResTy VPR64:$src),
              (OpTy OpGPR:$Rn),
              (OpImm:$Imm))),
        (ResTy (EXTRACT_SUBREG 
          (ExtResTy (INS (ExtResTy (SUBREG_TO_REG (i64 0), VPR64:$src, sub_64)),
            OpGPR:$Rn, OpImm:$Imm)), sub_64))>;

def INSbw_pattern : Neon_INS_main_pattern<v8i8, v16i8, GPR32, i32,
                                          neon_uimm3_bare, INSbw>;
def INShw_pattern : Neon_INS_main_pattern<v4i16, v8i16, GPR32, i32,
                                          neon_uimm2_bare, INShw>;
def INSsw_pattern : Neon_INS_main_pattern<v2i32, v4i32, GPR32, i32,
                                          neon_uimm1_bare, INSsw>;
def INSdx_pattern : Neon_INS_main_pattern<v1i64, v2i64, GPR64, i64,
                                          neon_uimm0_bare, INSdx>;

class NeonI_INS_element<string asmop, string Res, ValueType ResTy,
                        Operand ResImm, ValueType MidTy>
  : NeonI_insert<0b1, 0b1,
                 (outs VPR128:$Rd), (ins VPR128:$src, VPR128:$Rn, 
                 ResImm:$Immd, ResImm:$Immn),
                 asmop # "\t$Rd." # Res # "[$Immd], $Rn." # Res # "[$Immn]",
                 [(set (ResTy VPR128:$Rd),
                    (ResTy (vector_insert
                      (ResTy VPR128:$src),
                      (MidTy (vector_extract
                        (ResTy VPR128:$Rn),
                        (ResImm:$Immn))),
                      (ResImm:$Immd))))],
                 NoItinerary> {
  let Constraints = "$src = $Rd";
  bits<4> Immd;
  bits<4> Immn;
}

//Insert element (vector, from element)
def INSELb : NeonI_INS_element<"ins", "b", v16i8, neon_uimm4_bare, i32> {
  let Inst{20-16} = {Immd{3}, Immd{2}, Immd{1}, Immd{0}, 0b1};
  let Inst{14-11} = {Immn{3}, Immn{2}, Immn{1}, Immn{0}};
}
def INSELh : NeonI_INS_element<"ins", "h", v8i16, neon_uimm3_bare, i32> {
  let Inst{20-16} = {Immd{2}, Immd{1}, Immd{0}, 0b1, 0b0};
  let Inst{14-12} = {Immn{2}, Immn{1}, Immn{0}};
  // bit 11 is unspecified.
}
def INSELs : NeonI_INS_element<"ins", "s", v4i32, neon_uimm2_bare, i32> {
  let Inst{20-16} = {Immd{1}, Immd{0}, 0b1, 0b0, 0b0};
  let Inst{14-13} = {Immn{1}, Immn{0}};
  // bits 11-12 are unspecified.
}
def INSELd : NeonI_INS_element<"ins", "d", v2i64, neon_uimm1_bare, i64> {
  let Inst{20-16} = {Immd, 0b1, 0b0, 0b0, 0b0};
  let Inst{14} = Immn{0};
  // bits 11-13 are unspecified.
}

multiclass Neon_INS_elt_float_pattern<ValueType ResTy, ValueType NaTy,
                                      ValueType MidTy,
                                      RegisterClass OpFPR, Operand ResImm,
                                      SubRegIndex SubIndex, Instruction INS> {
def : Pat<(ResTy (vector_insert
          (ResTy VPR128:$src),
          (MidTy (vector_extract
            (ResTy VPR128:$Rn),
            (ResImm:$Immn))),
          (ResImm:$Immd))),
        (INS (ResTy VPR128:$src), (ResTy VPR128:$Rn),
          ResImm:$Immd, ResImm:$Immn)>;

def : Pat <(ResTy (vector_insert
             (ResTy VPR128:$src),
             (MidTy OpFPR:$Rn),
             (ResImm:$Imm))),
           (INS (ResTy VPR128:$src),
             (ResTy (SUBREG_TO_REG (i64 0), OpFPR:$Rn, SubIndex)),
             ResImm:$Imm,
             (i64 0))>;

def : Pat <(NaTy (vector_insert
             (NaTy VPR64:$src),
             (MidTy OpFPR:$Rn),
             (ResImm:$Imm))),
           (NaTy (EXTRACT_SUBREG 
             (ResTy (INS 
               (ResTy (SUBREG_TO_REG (i64 0), (NaTy VPR64:$src), sub_64)),
               (ResTy (SUBREG_TO_REG (i64 0), (MidTy OpFPR:$Rn), SubIndex)),
               ResImm:$Imm,
               (i64 0))),
             sub_64))>;
}

defm : Neon_INS_elt_float_pattern<v4f32, v2f32, f32, FPR32, neon_uimm2_bare,
                                  sub_32, INSELs>;
defm : Neon_INS_elt_float_pattern<v2f64, v1f64, f64, FPR64, neon_uimm1_bare,
                                  sub_64, INSELd>;

multiclass Neon_INS_elt_pattern <ValueType NaTy, Operand NaImm,
                                ValueType MidTy, ValueType StTy,
                                Operand StImm, Instruction INS> { 
def : Pat<(NaTy (vector_insert
            (NaTy VPR64:$src),
            (MidTy (vector_extract
              (StTy VPR128:$Rn),
              (StImm:$Immn))),
            (NaImm:$Immd))),
          (NaTy (EXTRACT_SUBREG
            (StTy (INS 
              (StTy (SUBREG_TO_REG (i64 0), VPR64:$src, sub_64)),
              (StTy VPR128:$Rn),
              NaImm:$Immd,
              StImm:$Immn)),
          sub_64))>;

def : Pat<(StTy (vector_insert
            (StTy VPR128:$src),
            (MidTy (vector_extract
              (NaTy VPR64:$Rn),
              (NaImm:$Immn))),
            (StImm:$Immd))),
          (StTy (INS 
            (StTy VPR128:$src),
            (StTy (SUBREG_TO_REG (i64 0), VPR64:$Rn, sub_64)),
            StImm:$Immd,
            NaImm:$Immn))>;

def : Pat<(NaTy (vector_insert
            (NaTy VPR64:$src),
            (MidTy (vector_extract
              (NaTy VPR64:$Rn),
              (NaImm:$Immn))),
            (NaImm:$Immd))),
          (NaTy (EXTRACT_SUBREG
            (StTy (INS 
              (StTy (SUBREG_TO_REG (i64 0), VPR64:$src, sub_64)),
              (StTy (SUBREG_TO_REG (i64 0), VPR64:$Rn, sub_64)),
              NaImm:$Immd,
              NaImm:$Immn)),
          sub_64))>;
}

defm : Neon_INS_elt_pattern<v8i8, neon_uimm3_bare, i32,
                            v16i8, neon_uimm4_bare, INSELb>;
defm : Neon_INS_elt_pattern<v4i16, neon_uimm2_bare, i32,
                            v8i16, neon_uimm3_bare, INSELh>;
defm : Neon_INS_elt_pattern<v2i32, neon_uimm1_bare, i32,
                            v4i32, neon_uimm2_bare, INSELs>;
defm : Neon_INS_elt_pattern<v1i64, neon_uimm0_bare, i64,
                            v2i64, neon_uimm1_bare, INSELd>;


class NeonI_SMOV<string asmop, string Res, bit Q,
                 ValueType OpTy, ValueType eleTy,
                 Operand OpImm, RegisterClass ResGPR, ValueType ResTy>
  : NeonI_copy<Q, 0b0, 0b0101,
               (outs ResGPR:$Rd), (ins VPR128:$Rn, OpImm:$Imm),
               asmop # "\t$Rd, $Rn." # Res # "[$Imm]",
               [(set (ResTy ResGPR:$Rd),
                 (ResTy (sext_inreg
                   (ResTy (vector_extract
                     (OpTy VPR128:$Rn), (OpImm:$Imm))),
                   eleTy)))],
               NoItinerary> {
  bits<4> Imm;
}

//Signed integer move (main, from element)
def SMOVwb : NeonI_SMOV<"smov", "b", 0b0, v16i8, i8, neon_uimm4_bare,
                        GPR32, i32> {
  let Inst{20-16} = {Imm{3}, Imm{2}, Imm{1}, Imm{0}, 0b1};
}
def SMOVwh : NeonI_SMOV<"smov", "h", 0b0, v8i16, i16, neon_uimm3_bare,
                        GPR32, i32> {
  let Inst{20-16} = {Imm{2}, Imm{1}, Imm{0}, 0b1, 0b0};
}
def SMOVxb : NeonI_SMOV<"smov", "b", 0b1, v16i8, i8, neon_uimm4_bare,
                        GPR64, i64> {
  let Inst{20-16} = {Imm{3}, Imm{2}, Imm{1}, Imm{0}, 0b1};
}
def SMOVxh : NeonI_SMOV<"smov", "h", 0b1, v8i16, i16, neon_uimm3_bare,
                        GPR64, i64> {
  let Inst{20-16} = {Imm{2}, Imm{1}, Imm{0}, 0b1, 0b0};
}
def SMOVxs : NeonI_SMOV<"smov", "s", 0b1, v4i32, i32, neon_uimm2_bare,
                        GPR64, i64> {
  let Inst{20-16} = {Imm{1}, Imm{0}, 0b1, 0b0, 0b0};
}

multiclass Neon_SMOVx_pattern <ValueType StTy, ValueType NaTy,
                               ValueType eleTy, Operand StImm,  Operand NaImm,
                               Instruction SMOVI> {
  def : Pat<(i64 (sext_inreg
              (i64 (anyext
                (i32 (vector_extract
                  (StTy VPR128:$Rn), (StImm:$Imm))))),
              eleTy)),
            (SMOVI VPR128:$Rn, StImm:$Imm)>;
  
  def : Pat<(i64 (sext
              (i32 (vector_extract
                (StTy VPR128:$Rn), (StImm:$Imm))))),
            (SMOVI VPR128:$Rn, StImm:$Imm)>;
  
  def : Pat<(i64 (sext_inreg
              (i64 (vector_extract
                (NaTy VPR64:$Rn), (NaImm:$Imm))),
              eleTy)),
            (SMOVI (StTy (SUBREG_TO_REG (i64 0), VPR64:$Rn, sub_64)),
              NaImm:$Imm)>;
  
  def : Pat<(i64 (sext_inreg
              (i64 (anyext
                (i32 (vector_extract
                  (NaTy VPR64:$Rn), (NaImm:$Imm))))),
              eleTy)),
            (SMOVI (StTy (SUBREG_TO_REG (i64 0), VPR64:$Rn, sub_64)),
              NaImm:$Imm)>;
  
  def : Pat<(i64 (sext
              (i32 (vector_extract
                (NaTy VPR64:$Rn), (NaImm:$Imm))))),
            (SMOVI (StTy (SUBREG_TO_REG (i64 0), VPR64:$Rn, sub_64)),
              NaImm:$Imm)>; 
}

defm : Neon_SMOVx_pattern<v16i8, v8i8, i8, neon_uimm4_bare,
                          neon_uimm3_bare, SMOVxb>;
defm : Neon_SMOVx_pattern<v8i16, v4i16, i16, neon_uimm3_bare,
                          neon_uimm2_bare, SMOVxh>;
defm : Neon_SMOVx_pattern<v4i32, v2i32, i32, neon_uimm2_bare,
                          neon_uimm1_bare, SMOVxs>;

class Neon_SMOVw_pattern <ValueType StTy, ValueType NaTy,
                          ValueType eleTy, Operand StImm,  Operand NaImm,
                          Instruction SMOVI>
  : Pat<(i32 (sext_inreg
          (i32 (vector_extract
            (NaTy VPR64:$Rn), (NaImm:$Imm))),
          eleTy)),
        (SMOVI (StTy (SUBREG_TO_REG (i64 0), VPR64:$Rn, sub_64)),
          NaImm:$Imm)>;

def : Neon_SMOVw_pattern<v16i8, v8i8, i8, neon_uimm4_bare,
                         neon_uimm3_bare, SMOVwb>;
def : Neon_SMOVw_pattern<v8i16, v4i16, i16, neon_uimm3_bare,
                         neon_uimm2_bare, SMOVwh>;

class NeonI_UMOV<string asmop, string Res, bit Q,
                 ValueType OpTy, Operand OpImm,
                 RegisterClass ResGPR, ValueType ResTy>
  : NeonI_copy<Q, 0b0, 0b0111,
               (outs ResGPR:$Rd), (ins VPR128:$Rn, OpImm:$Imm),
               asmop # "\t$Rd, $Rn." # Res # "[$Imm]",
               [(set (ResTy ResGPR:$Rd),
                  (ResTy (vector_extract
                    (OpTy VPR128:$Rn), (OpImm:$Imm))))],
               NoItinerary> {
  bits<4> Imm;
}

//Unsigned integer move (main, from element)
def UMOVwb : NeonI_UMOV<"umov", "b", 0b0, v16i8, neon_uimm4_bare,
                         GPR32, i32> {
  let Inst{20-16} = {Imm{3}, Imm{2}, Imm{1}, Imm{0}, 0b1};
}
def UMOVwh : NeonI_UMOV<"umov", "h", 0b0, v8i16, neon_uimm3_bare,
                         GPR32, i32> {
  let Inst{20-16} = {Imm{2}, Imm{1}, Imm{0}, 0b1, 0b0};
}
def UMOVws : NeonI_UMOV<"umov", "s", 0b0, v4i32, neon_uimm2_bare,
                         GPR32, i32> {
  let Inst{20-16} = {Imm{1}, Imm{0}, 0b1, 0b0, 0b0};
}
def UMOVxd : NeonI_UMOV<"umov", "d", 0b1, v2i64, neon_uimm1_bare,
                         GPR64, i64> {
  let Inst{20-16} = {Imm, 0b1, 0b0, 0b0, 0b0};
}

class Neon_UMOV_pattern <ValueType StTy, ValueType NaTy, ValueType ResTy,
                         Operand StImm,  Operand NaImm,
                         Instruction SMOVI>
  : Pat<(ResTy (vector_extract
          (NaTy VPR64:$Rn), NaImm:$Imm)),
        (SMOVI (StTy (SUBREG_TO_REG (i64 0), VPR64:$Rn, sub_64)),
          NaImm:$Imm)>;

def : Neon_UMOV_pattern<v16i8, v8i8, i32, neon_uimm4_bare,
                        neon_uimm3_bare, UMOVwb>;
def : Neon_UMOV_pattern<v8i16, v4i16, i32, neon_uimm3_bare,
                        neon_uimm2_bare, UMOVwh>; 
def : Neon_UMOV_pattern<v4i32, v2i32, i32, neon_uimm2_bare,
                        neon_uimm1_bare, UMOVws>;

def : Pat<(i32 (and
            (i32 (vector_extract
              (v16i8 VPR128:$Rn), (neon_uimm4_bare:$Imm))),
            255)),
          (UMOVwb VPR128:$Rn, neon_uimm4_bare:$Imm)>;

def : Pat<(i32 (and
            (i32 (vector_extract
              (v8i16 VPR128:$Rn), (neon_uimm3_bare:$Imm))),
            65535)),
          (UMOVwh VPR128:$Rn, neon_uimm3_bare:$Imm)>;

def : Pat<(i64 (zext
            (i32 (vector_extract
              (v2i64 VPR128:$Rn), (neon_uimm1_bare:$Imm))))),
          (UMOVxd VPR128:$Rn, neon_uimm1_bare:$Imm)>;

def : Pat<(i32 (and
            (i32 (vector_extract
              (v8i8 VPR64:$Rn), (neon_uimm3_bare:$Imm))),
            255)),
          (UMOVwb (SUBREG_TO_REG (i64 0), VPR64:$Rn, sub_64),
            neon_uimm3_bare:$Imm)>;

def : Pat<(i32 (and
            (i32 (vector_extract
              (v4i16 VPR64:$Rn), (neon_uimm2_bare:$Imm))),
            65535)),
          (UMOVwh (SUBREG_TO_REG (i64 0), VPR64:$Rn, sub_64),
            neon_uimm2_bare:$Imm)>;

def : Pat<(i64 (zext
            (i32 (vector_extract
              (v1i64 VPR64:$Rn), (neon_uimm0_bare:$Imm))))),
          (UMOVxd (SUBREG_TO_REG (i64 0), VPR64:$Rn, sub_64),
            neon_uimm0_bare:$Imm)>;

// Additional copy patterns for scalar types
def : Pat<(i32 (vector_extract (v1i8 FPR8:$Rn), (i64 0))),
          (UMOVwb (v16i8
            (SUBREG_TO_REG (i64 0), FPR8:$Rn, sub_8)), (i64 0))>;

def : Pat<(i32 (vector_extract (v1i16 FPR16:$Rn), (i64 0))),
          (UMOVwh (v8i16
            (SUBREG_TO_REG (i64 0), FPR16:$Rn, sub_16)), (i64 0))>;

def : Pat<(i32 (vector_extract (v1i32 FPR32:$Rn), (i64 0))),
          (FMOVws FPR32:$Rn)>;

def : Pat<(i64 (vector_extract (v1i64 FPR64:$Rn), (i64 0))),
          (FMOVxd FPR64:$Rn)>;
               
def : Pat<(f64 (vector_extract (v1f64 FPR64:$Rn), (i64 0))),
          (f64 FPR64:$Rn)>;

def : Pat<(f32 (vector_extract (v1f32 FPR32:$Rn), (i64 0))),
          (f32 FPR32:$Rn)>;

def : Pat<(v1i8 (scalar_to_vector GPR32:$Rn)),
          (v1i8 (EXTRACT_SUBREG (v16i8
            (INSbw (v16i8 (IMPLICIT_DEF)), $Rn, (i64 0))),
            sub_8))>;

def : Pat<(v1i16 (scalar_to_vector GPR32:$Rn)),
          (v1i16 (EXTRACT_SUBREG (v8i16
            (INShw (v8i16 (IMPLICIT_DEF)), $Rn, (i64 0))),
            sub_16))>;

def : Pat<(v1i32 (scalar_to_vector GPR32:$src)),
          (FMOVsw $src)>;

def : Pat<(v1i64 (scalar_to_vector GPR64:$src)),
          (FMOVdx $src)>;

def : Pat<(v1f32 (scalar_to_vector (f32 FPR32:$Rn))),
          (v1f32 FPR32:$Rn)>;
def : Pat<(v1f64 (scalar_to_vector (f64 FPR64:$Rn))),
          (v1f64 FPR64:$Rn)>;

def : Pat<(v1f64 (scalar_to_vector (f64 FPR64:$src))),
          (FMOVdd $src)>;

def : Pat<(v2f64 (scalar_to_vector (f64 FPR64:$src))),
          (INSERT_SUBREG (v2f64 (IMPLICIT_DEF)),
                         (f64 FPR64:$src), sub_64)>;

class NeonI_DUP_Elt<bit Q, string asmop, string rdlane,  string rnlane,
                    RegisterOperand ResVPR, ValueType ResTy,
                    ValueType OpTy, Operand OpImm>
  : NeonI_copy<Q, 0b0, 0b0000, (outs ResVPR:$Rd),
               (ins VPR128:$Rn, OpImm:$Imm),
               asmop # "\t$Rd" # rdlane # ", $Rn" # rnlane # "[$Imm]",
               [],
               NoItinerary> {
  bits<4> Imm;
}

def DUPELT16b : NeonI_DUP_Elt<0b1, "dup", ".16b", ".b", VPR128, v16i8, v16i8,
                              neon_uimm4_bare> {
  let Inst{20-16} = {Imm{3}, Imm{2}, Imm{1}, Imm{0}, 0b1};
}

def DUPELT8h : NeonI_DUP_Elt<0b1, "dup", ".8h", ".h", VPR128, v8i16, v8i16,
                              neon_uimm3_bare> {
  let Inst{20-16} = {Imm{2}, Imm{1}, Imm{0}, 0b1, 0b0};
}

def DUPELT4s : NeonI_DUP_Elt<0b1, "dup", ".4s", ".s", VPR128, v4i32, v4i32,
                              neon_uimm2_bare> {
  let Inst{20-16} = {Imm{1}, Imm{0}, 0b1, 0b0, 0b0};
}

def DUPELT2d : NeonI_DUP_Elt<0b1, "dup", ".2d", ".d", VPR128, v2i64, v2i64,
                              neon_uimm1_bare> {
  let Inst{20-16} = {Imm, 0b1, 0b0, 0b0, 0b0};
}

def DUPELT8b : NeonI_DUP_Elt<0b0, "dup", ".8b", ".b", VPR64, v8i8, v16i8,
                              neon_uimm4_bare> {
  let Inst{20-16} = {Imm{3}, Imm{2}, Imm{1}, Imm{0}, 0b1};
}

def DUPELT4h : NeonI_DUP_Elt<0b0, "dup", ".4h", ".h", VPR64, v4i16, v8i16,
                              neon_uimm3_bare> {
  let Inst{20-16} = {Imm{2}, Imm{1}, Imm{0}, 0b1, 0b0};
}

def DUPELT2s : NeonI_DUP_Elt<0b0, "dup", ".2s", ".s", VPR64, v2i32, v4i32,
                              neon_uimm2_bare> {
  let Inst{20-16} = {Imm{1}, Imm{0}, 0b1, 0b0, 0b0};
}

multiclass NeonI_DUP_Elt_pattern<Instruction DUPELT, ValueType ResTy,
                                       ValueType OpTy,ValueType NaTy,
                                       ValueType ExTy, Operand OpLImm,
                                       Operand OpNImm> {
def  : Pat<(ResTy (Neon_vduplane (OpTy VPR128:$Rn), OpLImm:$Imm)),
        (ResTy (DUPELT (OpTy VPR128:$Rn), OpLImm:$Imm))>;

def : Pat<(ResTy (Neon_vduplane
            (NaTy VPR64:$Rn), OpNImm:$Imm)),
          (ResTy (DUPELT
            (ExTy (SUBREG_TO_REG (i64 0), VPR64:$Rn, sub_64)), OpNImm:$Imm))>;
}
defm : NeonI_DUP_Elt_pattern<DUPELT16b, v16i8, v16i8, v8i8, v16i8,
                             neon_uimm4_bare, neon_uimm3_bare>;
defm : NeonI_DUP_Elt_pattern<DUPELT8b, v8i8, v16i8, v8i8, v16i8,
                             neon_uimm4_bare, neon_uimm3_bare>;
defm : NeonI_DUP_Elt_pattern<DUPELT8h, v8i16, v8i16, v4i16, v8i16,
                             neon_uimm3_bare, neon_uimm2_bare>;
defm : NeonI_DUP_Elt_pattern<DUPELT4h, v4i16, v8i16, v4i16, v8i16,
                             neon_uimm3_bare, neon_uimm2_bare>;
defm : NeonI_DUP_Elt_pattern<DUPELT4s, v4i32, v4i32, v2i32, v4i32,
                             neon_uimm2_bare, neon_uimm1_bare>;
defm : NeonI_DUP_Elt_pattern<DUPELT2s, v2i32, v4i32, v2i32, v4i32,
                             neon_uimm2_bare, neon_uimm1_bare>;
defm : NeonI_DUP_Elt_pattern<DUPELT2d, v2i64, v2i64, v1i64, v2i64,
                             neon_uimm1_bare, neon_uimm0_bare>;
defm : NeonI_DUP_Elt_pattern<DUPELT4s, v4f32, v4f32, v2f32, v4f32,
                             neon_uimm2_bare, neon_uimm1_bare>;
defm : NeonI_DUP_Elt_pattern<DUPELT2s, v2f32, v4f32, v2f32, v4f32,
                             neon_uimm2_bare, neon_uimm1_bare>;
defm : NeonI_DUP_Elt_pattern<DUPELT2d, v2f64, v2f64, v1f64, v2f64,
                             neon_uimm1_bare, neon_uimm0_bare>;

def : Pat<(v2f32 (Neon_vdup (f32 FPR32:$Rn))),
          (v2f32 (DUPELT2s 
            (SUBREG_TO_REG (i64 0), FPR32:$Rn, sub_32),
            (i64 0)))>;
def : Pat<(v4f32 (Neon_vdup (f32 FPR32:$Rn))),
          (v4f32 (DUPELT4s 
            (SUBREG_TO_REG (i64 0), FPR32:$Rn, sub_32),
            (i64 0)))>;
def : Pat<(v2f64 (Neon_vdup (f64 FPR64:$Rn))),
          (v2f64 (DUPELT2d 
            (SUBREG_TO_REG (i64 0), FPR64:$Rn, sub_64),
            (i64 0)))>;

class NeonI_DUP<bit Q, string asmop, string rdlane,
                RegisterOperand ResVPR, ValueType ResTy,
                RegisterClass OpGPR, ValueType OpTy>
  : NeonI_copy<Q, 0b0, 0b0001, (outs ResVPR:$Rd), (ins OpGPR:$Rn),
               asmop # "\t$Rd" # rdlane # ", $Rn",
               [(set (ResTy ResVPR:$Rd), 
                 (ResTy (Neon_vdup (OpTy OpGPR:$Rn))))],
               NoItinerary>;

def DUP16b : NeonI_DUP<0b1, "dup", ".16b", VPR128, v16i8, GPR32, i32> {
  let Inst{16} = 0b1;
  // bits 17-19 are unspecified.
}

def DUP8h : NeonI_DUP<0b1, "dup", ".8h", VPR128, v8i16, GPR32, i32> {
  let Inst{17-16} = 0b10;
  // bits 18-19 are unspecified.
}

def DUP4s : NeonI_DUP<0b1, "dup", ".4s", VPR128, v4i32, GPR32, i32> {
  let Inst{18-16} = 0b100;
  // bit 19 is unspecified.
}

def DUP2d : NeonI_DUP<0b1, "dup", ".2d", VPR128, v2i64, GPR64, i64> {
  let Inst{19-16} = 0b1000;
}

def DUP8b : NeonI_DUP<0b0, "dup", ".8b", VPR64, v8i8, GPR32, i32> {
  let Inst{16} = 0b1;
  // bits 17-19 are unspecified.
}

def DUP4h : NeonI_DUP<0b0, "dup", ".4h", VPR64, v4i16, GPR32, i32> {
  let Inst{17-16} = 0b10;
  // bits 18-19 are unspecified.
}

def DUP2s : NeonI_DUP<0b0, "dup", ".2s", VPR64, v2i32, GPR32, i32> {
  let Inst{18-16} = 0b100;
  // bit 19 is unspecified.
}

// patterns for CONCAT_VECTORS
multiclass Concat_Vector_Pattern<ValueType ResTy, ValueType OpTy> {
def : Pat<(ResTy (concat_vectors (OpTy VPR64:$Rn), undef)),
          (SUBREG_TO_REG (i64 0), VPR64:$Rn, sub_64)>;
def : Pat<(ResTy (concat_vectors (OpTy VPR64:$Rn), (OpTy VPR64:$Rm))),
          (INSELd 
            (v2i64 (SUBREG_TO_REG (i64 0), VPR64:$Rn, sub_64)),
            (v2i64 (SUBREG_TO_REG (i64 0), VPR64:$Rm, sub_64)),
            (i64 1),
            (i64 0))>;
def : Pat<(ResTy (concat_vectors (OpTy VPR64:$Rn), (OpTy VPR64:$Rn))),
          (DUPELT2d 
            (v2i64 (SUBREG_TO_REG (i64 0), VPR64:$Rn, sub_64)),
            (i64 0))> ;
}

defm : Concat_Vector_Pattern<v16i8, v8i8>;
defm : Concat_Vector_Pattern<v8i16, v4i16>;
defm : Concat_Vector_Pattern<v4i32, v2i32>;
defm : Concat_Vector_Pattern<v2i64, v1i64>;
defm : Concat_Vector_Pattern<v4f32, v2f32>;
defm : Concat_Vector_Pattern<v2f64, v1f64>;

//patterns for EXTRACT_SUBVECTOR
def : Pat<(v8i8 (extract_subvector (v16i8 VPR128:$Rn), (i64 0))),
          (v8i8 (EXTRACT_SUBREG VPR128:$Rn, sub_64))>;
def : Pat<(v4i16 (extract_subvector (v8i16 VPR128:$Rn), (i64 0))),
          (v4i16 (EXTRACT_SUBREG VPR128:$Rn, sub_64))>;
def : Pat<(v2i32 (extract_subvector (v4i32 VPR128:$Rn), (i64 0))),
          (v2i32 (EXTRACT_SUBREG VPR128:$Rn, sub_64))>;
def : Pat<(v1i64 (extract_subvector (v2i64 VPR128:$Rn), (i64 0))),
          (v1i64 (EXTRACT_SUBREG VPR128:$Rn, sub_64))>;
def : Pat<(v2f32 (extract_subvector (v4f32 VPR128:$Rn), (i64 0))),
          (v2f32 (EXTRACT_SUBREG VPR128:$Rn, sub_64))>;
def : Pat<(v1f64 (extract_subvector (v2f64 VPR128:$Rn), (i64 0))),
          (v1f64 (EXTRACT_SUBREG VPR128:$Rn, sub_64))>;
