//===- HexagonInstrInfoVector.td - Hexagon Vector Patterns -*- tablegen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file describes the Hexagon Vector instructions in TableGen format.
//
//===----------------------------------------------------------------------===//

def V2I1:  PatLeaf<(v2i1  PredRegs:$R)>;
def V4I1:  PatLeaf<(v4i1  PredRegs:$R)>;
def V8I1:  PatLeaf<(v8i1  PredRegs:$R)>;
def V4I8:  PatLeaf<(v4i8  IntRegs:$R)>;
def V2I16: PatLeaf<(v2i16 IntRegs:$R)>;
def V8I8:  PatLeaf<(v8i8  DoubleRegs:$R)>;
def V4I16: PatLeaf<(v4i16 DoubleRegs:$R)>;
def V2I32: PatLeaf<(v2i32 DoubleRegs:$R)>;

// Vector shift support. Vector shifting in Hexagon is rather different
// from internal representation of LLVM.
// LLVM assumes all shifts (in vector case) will have the form
// <VT> = SHL/SRA/SRL <VT> by <VT>
// while Hexagon has the following format:
// <VT> = SHL/SRA/SRL <VT> by <IT/i32>
// As a result, special care is needed to guarantee correctness and
// performance.
class vshift_v4i16<SDNode Op, string Str, bits<3>MajOp, bits<3>MinOp>
  : S_2OpInstImm<Str, MajOp, MinOp, u4Imm,
      [(set (v4i16 DoubleRegs:$dst),
            (Op (v4i16 DoubleRegs:$src1), u4ImmPred:$src2))]> {
  bits<4> src2;
  let Inst{11-8} = src2;
}

class vshift_v2i32<SDNode Op, string Str, bits<3>MajOp, bits<3>MinOp>
  : S_2OpInstImm<Str, MajOp, MinOp, u5Imm,
      [(set (v2i32 DoubleRegs:$dst),
            (Op (v2i32 DoubleRegs:$src1), u5ImmPred:$src2))]> {
  bits<5> src2;
  let Inst{12-8} = src2;
}

let isCodeGenOnly = 0 in {
def S2_asr_i_vw : vshift_v2i32<sra, "vasrw", 0b010, 0b000>;
def S2_lsr_i_vw : vshift_v2i32<srl, "vlsrw", 0b010, 0b001>;
def S2_asl_i_vw : vshift_v2i32<shl, "vaslw", 0b010, 0b010>;

def S2_asr_i_vh : vshift_v4i16<sra, "vasrh", 0b100, 0b000>;
def S2_lsr_i_vh : vshift_v4i16<srl, "vlsrh", 0b100, 0b001>;
def S2_asl_i_vh : vshift_v4i16<shl, "vaslh", 0b100, 0b010>;

// Vector shift words by register
def S2_asr_r_vw : T_S3op_shiftVect < "vasrw", 0b00, 0b00>;
def S2_lsr_r_vw : T_S3op_shiftVect < "vlsrw", 0b00, 0b01>;
def S2_asl_r_vw : T_S3op_shiftVect < "vaslw", 0b00, 0b10>;
def S2_lsl_r_vw : T_S3op_shiftVect < "vlslw", 0b00, 0b11>;

// Vector shift halfwords by register
def S2_asr_r_vh : T_S3op_shiftVect < "vasrh", 0b01, 0b00>;
def S2_lsr_r_vh : T_S3op_shiftVect < "vlsrh", 0b01, 0b01>;
def S2_asl_r_vh : T_S3op_shiftVect < "vaslh", 0b01, 0b10>;
def S2_lsl_r_vh : T_S3op_shiftVect < "vlslh", 0b01, 0b11>;
}
