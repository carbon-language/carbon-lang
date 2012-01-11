//====- X86InstrXOP.td - Describe the X86 Instruction Set --*- tablegen -*-====//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===-----------------------------------------------------------------------===//
//
// This file describes XOP (eXtended OPerations)
//
//===-----------------------------------------------------------------------===//

multiclass xop2op<bits<8> opc, string OpcodeStr, Intrinsic Int, PatFrag memop> {
  def rr : IXOP<opc, MRMSrcReg, (outs VR128:$dst), (ins VR128:$src),
           !strconcat(OpcodeStr, "\t{$src, $dst|$dst, $src}"),
           [(set VR128:$dst, (Int VR128:$src))]>, VEX;
  def rm : IXOP<opc, MRMSrcMem, (outs VR128:$dst), (ins f128mem:$src),
           !strconcat(OpcodeStr, "\t{$src, $dst|$dst, $src}"),
           [(set VR128:$dst, (Int (bitconvert (memop addr:$src))))]>, VEX;
}

let isAsmParserOnly = 1 in {
  defm VPHSUBWD  : xop2op<0xE2, "vphsubwd", int_x86_xop_vphsubwd, memopv2i64>;
  defm VPHSUBDQ  : xop2op<0xE3, "vphsubdq", int_x86_xop_vphsubdq, memopv2i64>;
  defm VPHSUBBW  : xop2op<0xE1, "vphsubbw", int_x86_xop_vphsubbw, memopv2i64>;
  defm VPHADDWQ  : xop2op<0xC7, "vphaddwq", int_x86_xop_vphaddwq, memopv2i64>;
  defm VPHADDWD  : xop2op<0xC6, "vphaddwd", int_x86_xop_vphaddwd, memopv2i64>;
  defm VPHADDUWQ : xop2op<0xD7, "vphadduwq", int_x86_xop_vphadduwq, memopv2i64>;
  defm VPHADDUWD : xop2op<0xD6, "vphadduwd", int_x86_xop_vphadduwd, memopv2i64>;
  defm VPHADDUDQ : xop2op<0xDB, "vphaddudq", int_x86_xop_vphaddudq, memopv2i64>;
  defm VPHADDUBW : xop2op<0xD1, "vphaddubw", int_x86_xop_vphaddubw, memopv2i64>;
  defm VPHADDUBQ : xop2op<0xD3, "vphaddubq", int_x86_xop_vphaddubq, memopv2i64>;
  defm VPHADDUBD : xop2op<0xD2, "vphaddubd", int_x86_xop_vphaddubd, memopv2i64>;
  defm VPHADDDQ  : xop2op<0xCB, "vphadddq", int_x86_xop_vphadddq, memopv2i64>;
  defm VPHADDBW  : xop2op<0xC1, "vphaddbw", int_x86_xop_vphaddbw, memopv2i64>;
  defm VPHADDBQ  : xop2op<0xC3, "vphaddbq", int_x86_xop_vphaddbq, memopv2i64>;
  defm VPHADDBD  : xop2op<0xC2, "vphaddbd", int_x86_xop_vphaddbd, memopv2i64>;
  defm VFRCZPS   : xop2op<0x80, "vfrczps", int_x86_xop_vfrcz_ps, memopv4f32>;
  defm VFRCZPD   : xop2op<0x81, "vfrczpd", int_x86_xop_vfrcz_pd, memopv2f64>;
}

// Scalar load 2 addr operand instructions
let Constraints = "$src1 = $dst" in {
multiclass xop2opsld<bits<8> opc, string OpcodeStr, Intrinsic Int,
                     Operand memop, ComplexPattern mem_cpat> {
  def rr : IXOP<opc, MRMSrcReg, (outs VR128:$dst), (ins VR128:$src1,
                                                        VR128:$src2),
           !strconcat(OpcodeStr, "\t{$src2, $dst|$dst, $src2}"),
           [(set VR128:$dst, (Int VR128:$src1, VR128:$src2))]>, VEX;
  def rm : IXOP<opc, MRMSrcMem, (outs VR128:$dst), (ins VR128:$src1,
                                                        memop:$src2),
           !strconcat(OpcodeStr, "\t{$src2, $dst|$dst, $src2}"),
           [(set VR128:$dst, (Int VR128:$src1,
                                  (bitconvert mem_cpat:$src2)))]>, VEX;
}

} // Constraints = "$src1 = $dst"

let isAsmParserOnly = 1 in {
  defm VFRCZSS   : xop2opsld<0x82, "vfrczss", int_x86_xop_vfrcz_ss,
                   ssmem, sse_load_f32>;
  defm VFRCZSD   : xop2opsld<0x83, "vfrczsd", int_x86_xop_vfrcz_sd,
                   sdmem, sse_load_f64>;
}


multiclass xop2op256<bits<8> opc, string OpcodeStr, Intrinsic Int,
                     PatFrag memop> {
  def rrY : IXOP<opc, MRMSrcReg, (outs VR256:$dst), (ins VR256:$src),
           !strconcat(OpcodeStr, "\t{$src, $dst|$dst, $src}"),
           [(set VR256:$dst, (Int VR256:$src))]>, VEX, VEX_L;
  def rmY : IXOP<opc, MRMSrcMem, (outs VR256:$dst), (ins f256mem:$src),
           !strconcat(OpcodeStr, "\t{$src, $dst|$dst, $src}"),
           [(set VR256:$dst, (Int (bitconvert (memop addr:$src))))]>, VEX;
}

let isAsmParserOnly = 1 in {
  defm VFRCZPS : xop2op256<0x80, "vfrczps", int_x86_xop_vfrcz_ps_256,
                           memopv8f32>;
  defm VFRCZPD : xop2op256<0x81, "vfrczpd", int_x86_xop_vfrcz_pd_256,
                           memopv4f64>;
}

multiclass xop3op<bits<8> opc, string OpcodeStr, Intrinsic Int> {
  def rr : IXOP<opc, MRMSrcReg, (outs VR128:$dst),
           (ins VR128:$src1, VR128:$src2),
           !strconcat(OpcodeStr, "\t{$src2, $src1, $dst|$dst, $src1, $src2}"),
           [(set VR128:$dst, (Int VR128:$src1, VR128:$src2))]>, VEX_4VOp3;
  def rm : IXOP<opc, MRMSrcMem, (outs VR128:$dst),
           (ins VR128:$src1, f128mem:$src2),
           !strconcat(OpcodeStr, "\t{$src2, $src1, $dst|$dst, $src1, $src2}"),
           [(set VR128:$dst,
              (Int VR128:$src1, (bitconvert (memopv2i64 addr:$src2))))]>,
           VEX_4V, VEX_W;
  def mr : IXOP<opc, MRMSrcMem, (outs VR128:$dst),
           (ins f128mem:$src1, VR128:$src2),
           !strconcat(OpcodeStr, "\t{$src2, $src1, $dst|$dst, $src1, $src2}"),
           [(set VR128:$dst,
              (Int (bitconvert (memopv2i64 addr:$src1)), VR128:$src2))]>,
             VEX_4VOp3;
}

let isAsmParserOnly = 1 in {
  defm VPSHLW : xop3op<0x95, "vpshlw", int_x86_xop_vpshlw>;
  defm VPSHLQ : xop3op<0x97, "vpshlq", int_x86_xop_vpshlq>;
  defm VPSHLD : xop3op<0x96, "vpshld", int_x86_xop_vpshld>;
  defm VPSHLB : xop3op<0x94, "vpshlb", int_x86_xop_vpshlb>;
  defm VPSHAW : xop3op<0x99, "vpshaw", int_x86_xop_vpshaw>;
  defm VPSHAQ : xop3op<0x9B, "vpshaq", int_x86_xop_vpshaq>;
  defm VPSHAD : xop3op<0x9A, "vpshad", int_x86_xop_vpshad>;
  defm VPSHAB : xop3op<0x98, "vpshab", int_x86_xop_vpshab>;
  defm VPROTW : xop3op<0x91, "vprotw", int_x86_xop_vprotw>;
  defm VPROTQ : xop3op<0x93, "vprotq", int_x86_xop_vprotq>;
  defm VPROTD : xop3op<0x92, "vprotd", int_x86_xop_vprotd>;
  defm VPROTB : xop3op<0x90, "vprotb", int_x86_xop_vprotb>;
}

multiclass xop3opimm<bits<8> opc, string OpcodeStr> {
  def ri : IXOPi8<opc, MRMSrcReg, (outs VR128:$dst),
           (ins VR128:$src1, i8imm:$src2),
           !strconcat(OpcodeStr, "\t{$src2, $src1, $dst|$dst, $src1, $src2}"),
           []>, VEX;
  def mi : IXOPi8<opc, MRMSrcMem, (outs VR128:$dst),
           (ins f128mem:$src1, i8imm:$src2),
           !strconcat(OpcodeStr, "\t{$src2, $src1, $dst|$dst, $src1, $src2}"),
           []>, VEX;
}

let isAsmParserOnly = 1 in {
  defm VPROTW : xop3opimm<0xC1, "vprotw">;
  defm VPROTQ : xop3opimm<0xC3, "vprotq">;
  defm VPROTD : xop3opimm<0xC2, "vprotd">;
  defm VPROTB : xop3opimm<0xC0, "vprotb">;
}

// Instruction where second source can be memory, but third must be register
multiclass xop4opm2<bits<8> opc, string OpcodeStr, Intrinsic Int> {
  def rr : IXOPi8<opc, MRMSrcReg, (outs VR128:$dst),
           (ins VR128:$src1, VR128:$src2, VR128:$src3),
           !strconcat(OpcodeStr,
           "\t{$src3, $src2, $src1, $dst|$dst, $src1, $src2, $src3}"),
           [(set VR128:$dst,
              (Int VR128:$src1, VR128:$src2, VR128:$src3))]>, VEX_4V, VEX_I8IMM;
  def rm : IXOPi8<opc, MRMSrcMem, (outs VR128:$dst),
           (ins VR128:$src1, f128mem:$src2, VR128:$src3),
           !strconcat(OpcodeStr,
           "\t{$src3, $src2, $src1, $dst|$dst, $src1, $src2, $src3}"),
           [(set VR128:$dst,
              (Int VR128:$src1, (bitconvert (memopv2i64 addr:$src2)),
              VR128:$src3))]>, VEX_4V, VEX_I8IMM;
}

let isAsmParserOnly = 1 in {
  defm VPMADCSWD  : xop4opm2<0xB6, "vpmadcswd", int_x86_xop_vpmadcswd>;
  defm VPMADCSSWD : xop4opm2<0xA6, "vpmadcsswd", int_x86_xop_vpmadcsswd>;
  defm VPMACSWW   : xop4opm2<0x95, "vpmacsww", int_x86_xop_vpmacsww>;
  defm VPMACSWD   : xop4opm2<0x96, "vpmacswd", int_x86_xop_vpmacswd>;
  defm VPMACSSWW  : xop4opm2<0x85, "vpmacssww", int_x86_xop_vpmacssww>;
  defm VPMACSSWD  : xop4opm2<0x86, "vpmacsswd", int_x86_xop_vpmacsswd>;
  defm VPMACSSDQL : xop4opm2<0x87, "vpmacssdql", int_x86_xop_vpmacssdql>;
  defm VPMACSSDQH : xop4opm2<0x8F, "vpmacssdqh", int_x86_xop_vpmacssdqh>;
  defm VPMACSSDD  : xop4opm2<0x8E, "vpmacssdd", int_x86_xop_vpmacssdd>;
  defm VPMACSDQL  : xop4opm2<0x97, "vpmacsdql", int_x86_xop_vpmacsdql>;
  defm VPMACSDQH  : xop4opm2<0x9F, "vpmacsdqh", int_x86_xop_vpmacsdqh>;
  defm VPMACSDD   : xop4opm2<0x9E, "vpmacsdd", int_x86_xop_vpmacsdd>;
}

// Instruction where second source can be memory, third must be imm8
multiclass xop4opimm<bits<8> opc, string OpcodeStr> {
  def ri : IXOPi8<opc, MRMSrcReg, (outs VR128:$dst),
           (ins VR128:$src1, VR128:$src2, i8imm:$src3),
           !strconcat(OpcodeStr,
           "\t{$src3, $src2, $src1, $dst|$dst, $src1, $src2, $src3}"),
           []>, VEX_4V;
  def mi : IXOPi8<opc, MRMSrcMem, (outs VR128:$dst),
           (ins VR128:$src1, f128mem:$src2, i8imm:$src3),
           !strconcat(OpcodeStr,
           "\t{$src3, $src2, $src1, $dst|$dst, $src1, $src2, $src3}"),
           []>, VEX_4V;
}

let isAsmParserOnly = 1 in {
  defm VPCOMW  : xop4opimm<0xCD, "vpcomw">;
  defm VPCOMUW : xop4opimm<0xED, "vpcomuw">;
  defm VPCOMUQ : xop4opimm<0xEF, "vpcomuq">;
  defm VPCOMUD : xop4opimm<0xEE, "vpcomud">;
  defm VPCOMUB : xop4opimm<0xEC, "vpcomub">;
  defm VPCOMQ  : xop4opimm<0xCF, "vpcomq">;
  defm VPCOMD  : xop4opimm<0xCE, "vpcomd">;
  defm VPCOMB  : xop4opimm<0xCC, "vpcomb">;
}

// Instruction where either second or third source can be memory
multiclass xop4op<bits<8> opc, string OpcodeStr> {
  def rr : IXOPi8<opc, MRMSrcReg, (outs VR128:$dst),
           (ins VR128:$src1, VR128:$src2, VR128:$src3),
           !strconcat(OpcodeStr,
           "\t{$src3, $src2, $src1, $dst|$dst, $src1, $src2, $src3}"),
           []>, VEX_4V, VEX_I8IMM;
  def rm : IXOPi8<opc, MRMSrcMem, (outs VR128:$dst),
           (ins VR128:$src1, VR128:$src2, f128mem:$src3),
           !strconcat(OpcodeStr,
           "\t{$src3, $src2, $src1, $dst|$dst, $src1, $src2, $src3}"),
           []>, VEX_4V, VEX_I8IMM, VEX_W, MemOp4;
  def mr : IXOPi8<opc, MRMSrcMem, (outs VR128:$dst),
           (ins VR128:$src1, f128mem:$src2, VR128:$src3),
           !strconcat(OpcodeStr,
           "\t{$src3, $src2, $src1, $dst|$dst, $src1, $src2, $src3}"),
           []>, VEX_4V, VEX_I8IMM;
}

let isAsmParserOnly = 1 in {
  defm VPPERM : xop4op<0xA3, "vpperm">;
  defm VPCMOV : xop4op<0xA2, "vpcmov">;
}

multiclass xop4op256<bits<8> opc, string OpcodeStr> {
  def rrY : IXOPi8<opc, MRMSrcReg, (outs VR256:$dst),
           (ins VR256:$src1, VR256:$src2, VR256:$src3),
           !strconcat(OpcodeStr,
           "\t{$src3, $src2, $src1, $dst|$dst, $src1, $src2, $src3}"),
           []>, VEX_4V, VEX_I8IMM;
  def rmY : IXOPi8<opc, MRMSrcMem, (outs VR256:$dst),
           (ins VR256:$src1, VR256:$src2, f256mem:$src3),
           !strconcat(OpcodeStr,
           "\t{$src3, $src2, $src1, $dst|$dst, $src1, $src2, $src3}"),
           []>, VEX_4V, VEX_I8IMM, VEX_W, MemOp4;
  def mrY : IXOPi8<opc, MRMSrcMem, (outs VR256:$dst),
           (ins VR256:$src1, f256mem:$src2, VR256:$src3),
           !strconcat(OpcodeStr,
           "\t{$src3, $src2, $src1, $dst|$dst, $src1, $src2, $src3}"),
           []>, VEX_4V, VEX_I8IMM;
}

let isAsmParserOnly = 1 in {
  defm VPCMOV : xop4op256<0xA2, "vpcmov">;
}

multiclass xop5op<bits<8> opc, string OpcodeStr, Intrinsic Int128,
                  Intrinsic Int256, PatFrag ld_128, PatFrag ld_256> {
  def rr : IXOP5<opc, MRMSrcReg, (outs VR128:$dst),
        (ins VR128:$src1, VR128:$src2, VR128:$src3, i8imm:$src4),
        !strconcat(OpcodeStr,
        "\t{$src4, $src3, $src2, $src1, $dst|$dst, $src1, $src2, $src3, $src4}"),
        [(set VR128:$dst,
           (Int128 VR128:$src1, VR128:$src2, VR128:$src3, imm:$src4))]>;
  def rm : IXOP5<opc, MRMSrcMem, (outs VR128:$dst),
        (ins VR128:$src1, VR128:$src2, f128mem:$src3, i8imm:$src4),
        !strconcat(OpcodeStr,
        "\t{$src4, $src3, $src2, $src1, $dst|$dst, $src1, $src2, $src3, $src4}"),
        [(set VR128:$dst,
           (Int128 VR128:$src1, VR128:$src2, (ld_128 addr:$src3), imm:$src4))]>,
        VEX_W, MemOp4;
  def mr : IXOP5<opc, MRMSrcMem, (outs VR128:$dst),
        (ins VR128:$src1, f128mem:$src2, VR128:$src3, i8imm:$src4),
        !strconcat(OpcodeStr,
        "\t{$src4, $src3, $src2, $src1, $dst|$dst, $src1, $src2, $src3, $src4}"),
        [(set VR128:$dst,
           (Int128 VR128:$src1, (ld_128 addr:$src2), VR128:$src3, imm:$src4))]>;
  def rrY : IXOP5<opc, MRMSrcReg, (outs VR256:$dst),
        (ins VR256:$src1, VR256:$src2, VR256:$src3, i8imm:$src4),
        !strconcat(OpcodeStr,
        "\t{$src4, $src3, $src2, $src1, $dst|$dst, $src1, $src2, $src3, $src4}"),
        [(set VR256:$dst,
          (Int256 VR256:$src1, VR256:$src2, VR256:$src3, imm:$src4))]>;
  def rmY : IXOP5<opc, MRMSrcMem, (outs VR256:$dst),
        (ins VR256:$src1, VR256:$src2, f256mem:$src3, i8imm:$src4),
        !strconcat(OpcodeStr,
        "\t{$src4, $src3, $src2, $src1, $dst|$dst, $src1, $src2, $src3, $src4}"),
        [(set VR256:$dst,
          (Int256 VR256:$src1, VR256:$src2, (ld_256 addr:$src3), imm:$src4))]>,
        VEX_W, MemOp4;
  def mrY : IXOP5<opc, MRMSrcMem, (outs VR256:$dst),
        (ins VR256:$src1, f256mem:$src2, VR256:$src3, i8imm:$src4),
        !strconcat(OpcodeStr,
        "\t{$src4, $src3, $src2, $src1, $dst|$dst, $src1, $src2, $src3, $src4}"),
        [(set VR256:$dst,
           (Int256 VR256:$src1, (ld_256 addr:$src2), VR256:$src3, imm:$src4))]>;
}

defm VPERMIL2PD : xop5op<0x49, "vpermil2pd", int_x86_xop_vpermil2pd,
                         int_x86_xop_vpermil2pd_256, memopv2f64, memopv4f64>;
defm VPERMIL2PS : xop5op<0x48, "vpermil2ps", int_x86_xop_vpermil2ps,
                         int_x86_xop_vpermil2ps_256, memopv4f32, memopv8f32>;

// XOP Intrinsics patterns

// VPCOM EQ
def : Pat<(int_x86_xop_vpcomeqw VR128:$src1, VR128:$src2),
          (VPCOMWri VR128:$src1, VR128:$src2, (i8 4))>;
def : Pat<(int_x86_xop_vpcomeqw VR128:$src1,
          (bitconvert (memopv2i64 addr:$src2))),
          (VPCOMWmi VR128:$src1, addr:$src2, (i8 4))>;

def : Pat<(int_x86_xop_vpcomequw VR128:$src1, VR128:$src2),
          (VPCOMUWri VR128:$src1, VR128:$src2, (i8 4))>;
def : Pat<(int_x86_xop_vpcomequw VR128:$src1,
          (bitconvert (memopv2i64 addr:$src2))),
          (VPCOMUWmi VR128:$src1, addr:$src2, (i8 4))>;

def : Pat<(int_x86_xop_vpcomequq VR128:$src1, VR128:$src2),
          (VPCOMUQri VR128:$src1, VR128:$src2, (i8 4))>;
def : Pat<(int_x86_xop_vpcomequq VR128:$src1,
          (bitconvert (memopv2i64 addr:$src2))),
          (VPCOMUQmi VR128:$src1, addr:$src2, (i8 4))>;

def : Pat<(int_x86_xop_vpcomequd VR128:$src1, VR128:$src2),
          (VPCOMUDri VR128:$src1, VR128:$src2, (i8 4))>;
def : Pat<(int_x86_xop_vpcomequd VR128:$src1,
          (bitconvert (memopv2i64 addr:$src2))),
          (VPCOMUDmi VR128:$src1, addr:$src2, (i8 4))>;

def : Pat<(int_x86_xop_vpcomequb VR128:$src1, VR128:$src2),
          (VPCOMUBri VR128:$src1, VR128:$src2, (i8 4))>;
def : Pat<(int_x86_xop_vpcomequb VR128:$src1,
          (bitconvert (memopv2i64 addr:$src2))),
          (VPCOMUBmi VR128:$src1, addr:$src2, (i8 4))>;

def : Pat<(int_x86_xop_vpcomeqq VR128:$src1, VR128:$src2),
          (VPCOMQri VR128:$src1, VR128:$src2, (i8 4))>;
def : Pat<(int_x86_xop_vpcomeqq VR128:$src1,
          (bitconvert (memopv2i64 addr:$src2))),
          (VPCOMQmi VR128:$src1, addr:$src2, (i8 4))>;

def : Pat<(int_x86_xop_vpcomeqd VR128:$src1, VR128:$src2),
          (VPCOMDri VR128:$src1, VR128:$src2, (i8 4))>;
def : Pat<(int_x86_xop_vpcomeqd VR128:$src1,
          (bitconvert (memopv2i64 addr:$src2))),
          (VPCOMDmi VR128:$src1, addr:$src2, (i8 4))>;

def : Pat<(int_x86_xop_vpcomeqb VR128:$src1, VR128:$src2),
          (VPCOMBri VR128:$src1, VR128:$src2, (i8 4))>;
def : Pat<(int_x86_xop_vpcomeqb VR128:$src1,
          (bitconvert (memopv2i64 addr:$src2))),
          (VPCOMBmi VR128:$src1, addr:$src2, (i8 4))>;

// VPCOM FALSE
def : Pat<(int_x86_xop_vpcomfalsew VR128:$src1, VR128:$src2),
          (VPCOMWri VR128:$src1, VR128:$src2, (i8 6))>;
def : Pat<(int_x86_xop_vpcomfalsew VR128:$src1,
          (bitconvert (memopv2i64 addr:$src2))),
          (VPCOMWmi VR128:$src1, addr:$src2, (i8 6))>;

def : Pat<(int_x86_xop_vpcomfalseuw VR128:$src1, VR128:$src2),
          (VPCOMUWri VR128:$src1, VR128:$src2, (i8 6))>;
def : Pat<(int_x86_xop_vpcomfalseuw VR128:$src1,
          (bitconvert (memopv2i64 addr:$src2))),
          (VPCOMUWmi VR128:$src1, addr:$src2, (i8 6))>;

def : Pat<(int_x86_xop_vpcomfalseuq VR128:$src1, VR128:$src2),
          (VPCOMUQri VR128:$src1, VR128:$src2, (i8 6))>;
def : Pat<(int_x86_xop_vpcomfalseuq VR128:$src1,
          (bitconvert (memopv2i64 addr:$src2))),
          (VPCOMUQmi VR128:$src1, addr:$src2, (i8 6))>;

def : Pat<(int_x86_xop_vpcomfalseud VR128:$src1, VR128:$src2),
          (VPCOMUDri VR128:$src1, VR128:$src2, (i8 6))>;
def : Pat<(int_x86_xop_vpcomfalseud VR128:$src1,
          (bitconvert (memopv2i64 addr:$src2))),
          (VPCOMUDmi VR128:$src1, addr:$src2, (i8 6))>;

def : Pat<(int_x86_xop_vpcomfalseub VR128:$src1, VR128:$src2),
          (VPCOMUBri VR128:$src1, VR128:$src2, (i8 6))>;
def : Pat<(int_x86_xop_vpcomfalseub VR128:$src1,
          (bitconvert (memopv2i64 addr:$src2))),
          (VPCOMUBmi VR128:$src1, addr:$src2, (i8 6))>;

def : Pat<(int_x86_xop_vpcomfalseq VR128:$src1, VR128:$src2),
          (VPCOMQri VR128:$src1, VR128:$src2, (i8 6))>;
def : Pat<(int_x86_xop_vpcomfalseq VR128:$src1,
          (bitconvert (memopv2i64 addr:$src2))),
          (VPCOMQmi VR128:$src1, addr:$src2, (i8 6))>;

def : Pat<(int_x86_xop_vpcomfalsed VR128:$src1, VR128:$src2),
          (VPCOMDri VR128:$src1, VR128:$src2, (i8 6))>;
def : Pat<(int_x86_xop_vpcomfalsed VR128:$src1,
          (bitconvert (memopv2i64 addr:$src2))),
          (VPCOMDmi VR128:$src1, addr:$src2, (i8 6))>;

def : Pat<(int_x86_xop_vpcomfalseb VR128:$src1, VR128:$src2),
          (VPCOMBri VR128:$src1, VR128:$src2, (i8 6))>;
def : Pat<(int_x86_xop_vpcomfalseb VR128:$src1,
          (bitconvert (memopv2i64 addr:$src2))),
          (VPCOMBmi VR128:$src1, addr:$src2, (i8 6))>;

// VPCOM GE
def : Pat<(int_x86_xop_vpcomgew VR128:$src1, VR128:$src2),
          (VPCOMWri VR128:$src1, VR128:$src2, (i8 3))>;
def : Pat<(int_x86_xop_vpcomgew VR128:$src1,
          (bitconvert (memopv2i64 addr:$src2))),
          (VPCOMWmi VR128:$src1, addr:$src2, (i8 3))>;

def : Pat<(int_x86_xop_vpcomgeuw VR128:$src1, VR128:$src2),
          (VPCOMUWri VR128:$src1, VR128:$src2, (i8 3))>;
def : Pat<(int_x86_xop_vpcomgeuw VR128:$src1,
          (bitconvert (memopv2i64 addr:$src2))),
          (VPCOMUWmi VR128:$src1, addr:$src2, (i8 3))>;

def : Pat<(int_x86_xop_vpcomgeuq VR128:$src1, VR128:$src2),
          (VPCOMUQri VR128:$src1, VR128:$src2, (i8 3))>;
def : Pat<(int_x86_xop_vpcomgeuq VR128:$src1,
          (bitconvert (memopv2i64 addr:$src2))),
          (VPCOMUQmi VR128:$src1, addr:$src2, (i8 3))>;

def : Pat<(int_x86_xop_vpcomgeud VR128:$src1, VR128:$src2),
          (VPCOMUDri VR128:$src1, VR128:$src2, (i8 3))>;
def : Pat<(int_x86_xop_vpcomgeud VR128:$src1,
          (bitconvert (memopv2i64 addr:$src2))),
          (VPCOMUDmi VR128:$src1, addr:$src2, (i8 3))>;

def : Pat<(int_x86_xop_vpcomgeub VR128:$src1, VR128:$src2),
          (VPCOMUBri VR128:$src1, VR128:$src2, (i8 3))>;
def : Pat<(int_x86_xop_vpcomgeub VR128:$src1,
          (bitconvert (memopv2i64 addr:$src2))),
          (VPCOMUBmi VR128:$src1, addr:$src2, (i8 3))>;

def : Pat<(int_x86_xop_vpcomgeq VR128:$src1, VR128:$src2),
          (VPCOMQri VR128:$src1, VR128:$src2, (i8 3))>;
def : Pat<(int_x86_xop_vpcomgeq VR128:$src1,
          (bitconvert (memopv2i64 addr:$src2))),
          (VPCOMQmi VR128:$src1, addr:$src2, (i8 3))>;

def : Pat<(int_x86_xop_vpcomged VR128:$src1, VR128:$src2),
          (VPCOMDri VR128:$src1, VR128:$src2, (i8 3))>;
def : Pat<(int_x86_xop_vpcomged VR128:$src1,
          (bitconvert (memopv2i64 addr:$src2))),
          (VPCOMDmi VR128:$src1, addr:$src2, (i8 3))>;

def : Pat<(int_x86_xop_vpcomgeb VR128:$src1, VR128:$src2),
          (VPCOMBri VR128:$src1, VR128:$src2, (i8 3))>;
def : Pat<(int_x86_xop_vpcomgeb VR128:$src1,
          (bitconvert (memopv2i64 addr:$src2))),
          (VPCOMBmi VR128:$src1, addr:$src2, (i8 3))>;

// VPCOM GT
def : Pat<(int_x86_xop_vpcomgtw VR128:$src1, VR128:$src2),
          (VPCOMWri VR128:$src1, VR128:$src2, (i8 2))>;
def : Pat<(int_x86_xop_vpcomgtw VR128:$src1,
          (bitconvert (memopv2i64 addr:$src2))),
          (VPCOMWmi VR128:$src1, addr:$src2, (i8 2))>;

def : Pat<(int_x86_xop_vpcomgtuw VR128:$src1, VR128:$src2),
          (VPCOMUWri VR128:$src1, VR128:$src2, (i8 2))>;
def : Pat<(int_x86_xop_vpcomgtuw VR128:$src1,
          (bitconvert (memopv2i64 addr:$src2))),
          (VPCOMUWmi VR128:$src1, addr:$src2, (i8 2))>;

def : Pat<(int_x86_xop_vpcomgtuq VR128:$src1, VR128:$src2),
          (VPCOMUQri VR128:$src1, VR128:$src2, (i8 2))>;
def : Pat<(int_x86_xop_vpcomgtuq VR128:$src1,
          (bitconvert (memopv2i64 addr:$src2))),
          (VPCOMUQmi VR128:$src1, addr:$src2, (i8 2))>;

def : Pat<(int_x86_xop_vpcomgtud VR128:$src1, VR128:$src2),
          (VPCOMUDri VR128:$src1, VR128:$src2, (i8 2))>;
def : Pat<(int_x86_xop_vpcomgtud VR128:$src1,
          (bitconvert (memopv2i64 addr:$src2))),
          (VPCOMUDmi VR128:$src1, addr:$src2, (i8 2))>;

def : Pat<(int_x86_xop_vpcomgtub VR128:$src1, VR128:$src2),
          (VPCOMUBri VR128:$src1, VR128:$src2, (i8 2))>;
def : Pat<(int_x86_xop_vpcomgtub VR128:$src1,
          (bitconvert (memopv2i64 addr:$src2))),
          (VPCOMUBmi VR128:$src1, addr:$src2, (i8 2))>;

def : Pat<(int_x86_xop_vpcomgtq VR128:$src1, VR128:$src2),
          (VPCOMQri VR128:$src1, VR128:$src2, (i8 2))>;
def : Pat<(int_x86_xop_vpcomgtq VR128:$src1,
          (bitconvert (memopv2i64 addr:$src2))),
          (VPCOMQmi VR128:$src1, addr:$src2, (i8 2))>;

def : Pat<(int_x86_xop_vpcomgtd VR128:$src1, VR128:$src2),
          (VPCOMDri VR128:$src1, VR128:$src2, (i8 2))>;
def : Pat<(int_x86_xop_vpcomgtd VR128:$src1,
          (bitconvert (memopv2i64 addr:$src2))),
          (VPCOMDmi VR128:$src1, addr:$src2, (i8 2))>;

def : Pat<(int_x86_xop_vpcomgtb VR128:$src1, VR128:$src2),
          (VPCOMBri VR128:$src1, VR128:$src2, (i8 2))>;
def : Pat<(int_x86_xop_vpcomgtb VR128:$src1,
          (bitconvert (memopv2i64 addr:$src2))),
          (VPCOMBmi VR128:$src1, addr:$src2, (i8 2))>;

// VPCOM LE
def : Pat<(int_x86_xop_vpcomlew VR128:$src1, VR128:$src2),
          (VPCOMWri VR128:$src1, VR128:$src2, (i8 1))>;
def : Pat<(int_x86_xop_vpcomlew VR128:$src1,
          (bitconvert (memopv2i64 addr:$src2))),
          (VPCOMWmi VR128:$src1, addr:$src2, (i8 1))>;

def : Pat<(int_x86_xop_vpcomleuw VR128:$src1, VR128:$src2),
          (VPCOMUWri VR128:$src1, VR128:$src2, (i8 1))>;
def : Pat<(int_x86_xop_vpcomleuw VR128:$src1,
          (bitconvert (memopv2i64 addr:$src2))),
          (VPCOMUWmi VR128:$src1, addr:$src2, (i8 1))>;

def : Pat<(int_x86_xop_vpcomleuq VR128:$src1, VR128:$src2),
          (VPCOMUQri VR128:$src1, VR128:$src2, (i8 1))>;
def : Pat<(int_x86_xop_vpcomleuq VR128:$src1,
          (bitconvert (memopv2i64 addr:$src2))),
          (VPCOMUQmi VR128:$src1, addr:$src2, (i8 1))>;

def : Pat<(int_x86_xop_vpcomleud VR128:$src1, VR128:$src2),
          (VPCOMUDri VR128:$src1, VR128:$src2, (i8 1))>;
def : Pat<(int_x86_xop_vpcomleud VR128:$src1,
          (bitconvert (memopv2i64 addr:$src2))),
          (VPCOMUDmi VR128:$src1, addr:$src2, (i8 1))>;

def : Pat<(int_x86_xop_vpcomleub VR128:$src1, VR128:$src2),
          (VPCOMUBri VR128:$src1, VR128:$src2, (i8 1))>;
def : Pat<(int_x86_xop_vpcomleub VR128:$src1,
          (bitconvert (memopv2i64 addr:$src2))),
          (VPCOMUBmi VR128:$src1, addr:$src2, (i8 1))>;

def : Pat<(int_x86_xop_vpcomleq VR128:$src1, VR128:$src2),
          (VPCOMQri VR128:$src1, VR128:$src2, (i8 1))>;
def : Pat<(int_x86_xop_vpcomleq VR128:$src1,
          (bitconvert (memopv2i64 addr:$src2))),
          (VPCOMQmi VR128:$src1, addr:$src2, (i8 1))>;

def : Pat<(int_x86_xop_vpcomled VR128:$src1, VR128:$src2),
          (VPCOMDri VR128:$src1, VR128:$src2, (i8 1))>;
def : Pat<(int_x86_xop_vpcomled VR128:$src1,
          (bitconvert (memopv2i64 addr:$src2))),
          (VPCOMDmi VR128:$src1, addr:$src2, (i8 1))>;

def : Pat<(int_x86_xop_vpcomleb VR128:$src1, VR128:$src2),
          (VPCOMBri VR128:$src1, VR128:$src2, (i8 1))>;
def : Pat<(int_x86_xop_vpcomleb VR128:$src1,
          (bitconvert (memopv2i64 addr:$src2))),
          (VPCOMBmi VR128:$src1, addr:$src2, (i8 1))>;

// VPCOM LT
def : Pat<(int_x86_xop_vpcomltw VR128:$src1, VR128:$src2),
          (VPCOMWri VR128:$src1, VR128:$src2, (i8 0))>;
def : Pat<(int_x86_xop_vpcomltw VR128:$src1,
          (bitconvert (memopv2i64 addr:$src2))),
          (VPCOMWmi VR128:$src1, addr:$src2, (i8 0))>;

def : Pat<(int_x86_xop_vpcomltuw VR128:$src1, VR128:$src2),
          (VPCOMUWri VR128:$src1, VR128:$src2, (i8 0))>;
def : Pat<(int_x86_xop_vpcomltuw VR128:$src1,
          (bitconvert (memopv2i64 addr:$src2))),
          (VPCOMUWmi VR128:$src1, addr:$src2, (i8 0))>;

def : Pat<(int_x86_xop_vpcomltuq VR128:$src1, VR128:$src2),
          (VPCOMUQri VR128:$src1, VR128:$src2, (i8 0))>;
def : Pat<(int_x86_xop_vpcomltuq VR128:$src1,
          (bitconvert (memopv2i64 addr:$src2))),
          (VPCOMUQmi VR128:$src1, addr:$src2, (i8 0))>;

def : Pat<(int_x86_xop_vpcomltud VR128:$src1, VR128:$src2),
          (VPCOMUDri VR128:$src1, VR128:$src2, (i8 0))>;
def : Pat<(int_x86_xop_vpcomltud VR128:$src1,
          (bitconvert (memopv2i64 addr:$src2))),
          (VPCOMUDmi VR128:$src1, addr:$src2, (i8 0))>;

def : Pat<(int_x86_xop_vpcomltub VR128:$src1, VR128:$src2),
          (VPCOMUBri VR128:$src1, VR128:$src2, (i8 0))>;
def : Pat<(int_x86_xop_vpcomltub VR128:$src1,
          (bitconvert (memopv2i64 addr:$src2))),
          (VPCOMUBmi VR128:$src1, addr:$src2, (i8 0))>;

def : Pat<(int_x86_xop_vpcomltq VR128:$src1, VR128:$src2),
          (VPCOMQri VR128:$src1, VR128:$src2, (i8 0))>;
def : Pat<(int_x86_xop_vpcomltq VR128:$src1,
          (bitconvert (memopv2i64 addr:$src2))),
          (VPCOMQmi VR128:$src1, addr:$src2, (i8 0))>;

def : Pat<(int_x86_xop_vpcomltd VR128:$src1, VR128:$src2),
          (VPCOMDri VR128:$src1, VR128:$src2, (i8 0))>;
def : Pat<(int_x86_xop_vpcomltd VR128:$src1,
          (bitconvert (memopv2i64 addr:$src2))),
          (VPCOMDmi VR128:$src1, addr:$src2, (i8 0))>;

def : Pat<(int_x86_xop_vpcomltb VR128:$src1, VR128:$src2),
          (VPCOMBri VR128:$src1, VR128:$src2, (i8 0))>;
def : Pat<(int_x86_xop_vpcomltb VR128:$src1,
          (bitconvert (memopv2i64 addr:$src2))),
          (VPCOMBmi VR128:$src1, addr:$src2, (i8 0))>;

// VPCOM NE
def : Pat<(int_x86_xop_vpcomnew VR128:$src1, VR128:$src2),
          (VPCOMWri VR128:$src1, VR128:$src2, (i8 5))>;
def : Pat<(int_x86_xop_vpcomnew VR128:$src1,
          (bitconvert (memopv2i64 addr:$src2))),
          (VPCOMWmi VR128:$src1, addr:$src2, (i8 5))>;

def : Pat<(int_x86_xop_vpcomneuw VR128:$src1, VR128:$src2),
          (VPCOMUWri VR128:$src1, VR128:$src2, (i8 5))>;
def : Pat<(int_x86_xop_vpcomneuw VR128:$src1,
          (bitconvert (memopv2i64 addr:$src2))),
          (VPCOMUWmi VR128:$src1, addr:$src2, (i8 5))>;

def : Pat<(int_x86_xop_vpcomneuq VR128:$src1, VR128:$src2),
          (VPCOMUQri VR128:$src1, VR128:$src2, (i8 5))>;
def : Pat<(int_x86_xop_vpcomneuq VR128:$src1,
          (bitconvert (memopv2i64 addr:$src2))),
          (VPCOMUQmi VR128:$src1, addr:$src2, (i8 5))>;

def : Pat<(int_x86_xop_vpcomneud VR128:$src1, VR128:$src2),
          (VPCOMUDri VR128:$src1, VR128:$src2, (i8 5))>;
def : Pat<(int_x86_xop_vpcomneud VR128:$src1,
          (bitconvert (memopv2i64 addr:$src2))),
          (VPCOMUDmi VR128:$src1, addr:$src2, (i8 5))>;

def : Pat<(int_x86_xop_vpcomneub VR128:$src1, VR128:$src2),
          (VPCOMUBri VR128:$src1, VR128:$src2, (i8 5))>;
def : Pat<(int_x86_xop_vpcomneub VR128:$src1,
          (bitconvert (memopv2i64 addr:$src2))),
          (VPCOMUBmi VR128:$src1, addr:$src2, (i8 5))>;

def : Pat<(int_x86_xop_vpcomneq VR128:$src1, VR128:$src2),
          (VPCOMQri VR128:$src1, VR128:$src2, (i8 5))>;
def : Pat<(int_x86_xop_vpcomneq VR128:$src1,
          (bitconvert (memopv2i64 addr:$src2))),
          (VPCOMQmi VR128:$src1, addr:$src2, (i8 5))>;

def : Pat<(int_x86_xop_vpcomned VR128:$src1, VR128:$src2),
          (VPCOMDri VR128:$src1, VR128:$src2, (i8 5))>;
def : Pat<(int_x86_xop_vpcomned VR128:$src1,
          (bitconvert (memopv2i64 addr:$src2))),
          (VPCOMDmi VR128:$src1, addr:$src2, (i8 5))>;

def : Pat<(int_x86_xop_vpcomneb VR128:$src1, VR128:$src2),
          (VPCOMBri VR128:$src1, VR128:$src2, (i8 5))>;
def : Pat<(int_x86_xop_vpcomneb VR128:$src1,
          (bitconvert (memopv2i64 addr:$src2))),
          (VPCOMBmi VR128:$src1, addr:$src2, (i8 5))>;

// VPCOM TRUE
def : Pat<(int_x86_xop_vpcomtruew VR128:$src1, VR128:$src2),
          (VPCOMWri VR128:$src1, VR128:$src2, (i8 6))>;
def : Pat<(int_x86_xop_vpcomtruew VR128:$src1,
          (bitconvert (memopv2i64 addr:$src2))),
          (VPCOMWmi VR128:$src1, addr:$src2, (i8 6))>;

def : Pat<(int_x86_xop_vpcomtrueuw VR128:$src1, VR128:$src2),
          (VPCOMUWri VR128:$src1, VR128:$src2, (i8 6))>;
def : Pat<(int_x86_xop_vpcomtrueuw VR128:$src1,
          (bitconvert (memopv2i64 addr:$src2))),
          (VPCOMUWmi VR128:$src1, addr:$src2, (i8 6))>;

def : Pat<(int_x86_xop_vpcomtrueuq VR128:$src1, VR128:$src2),
          (VPCOMUQri VR128:$src1, VR128:$src2, (i8 6))>;
def : Pat<(int_x86_xop_vpcomtrueuq VR128:$src1,
          (bitconvert (memopv2i64 addr:$src2))),
          (VPCOMUQmi VR128:$src1, addr:$src2, (i8 6))>;

def : Pat<(int_x86_xop_vpcomtrueud VR128:$src1, VR128:$src2),
          (VPCOMUDri VR128:$src1, VR128:$src2, (i8 6))>;
def : Pat<(int_x86_xop_vpcomtrueud VR128:$src1,
          (bitconvert (memopv2i64 addr:$src2))),
          (VPCOMUDmi VR128:$src1, addr:$src2, (i8 6))>;

def : Pat<(int_x86_xop_vpcomtrueub VR128:$src1, VR128:$src2),
          (VPCOMUBri VR128:$src1, VR128:$src2, (i8 6))>;
def : Pat<(int_x86_xop_vpcomtrueub VR128:$src1,
          (bitconvert (memopv2i64 addr:$src2))),
          (VPCOMUBmi VR128:$src1, addr:$src2, (i8 6))>;

def : Pat<(int_x86_xop_vpcomtrueq VR128:$src1, VR128:$src2),
          (VPCOMQri VR128:$src1, VR128:$src2, (i8 6))>;
def : Pat<(int_x86_xop_vpcomtrueq VR128:$src1,
          (bitconvert (memopv2i64 addr:$src2))),
          (VPCOMQmi VR128:$src1, addr:$src2, (i8 6))>;

def : Pat<(int_x86_xop_vpcomtrued VR128:$src1, VR128:$src2),
          (VPCOMDri VR128:$src1, VR128:$src2, (i8 6))>;
def : Pat<(int_x86_xop_vpcomtrued VR128:$src1,
          (bitconvert (memopv2i64 addr:$src2))),
          (VPCOMDmi VR128:$src1, addr:$src2, (i8 6))>;

def : Pat<(int_x86_xop_vpcomtrueb VR128:$src1, VR128:$src2),
          (VPCOMBri VR128:$src1, VR128:$src2, (i8 6))>;
def : Pat<(int_x86_xop_vpcomtrueb VR128:$src1,
          (bitconvert (memopv2i64 addr:$src2))),
          (VPCOMBmi VR128:$src1, addr:$src2, (i8 6))>;

// VPPERM
def : Pat<(int_x86_xop_vpperm VR128:$src1, VR128:$src2, VR128:$src3),
          (VPPERMrr VR128:$src1, VR128:$src2, VR128:$src3)>;
def : Pat<(int_x86_xop_vpperm VR128:$src1, VR128:$src2,
                                  (bitconvert (memopv2i64 addr:$src3))),
          (VPPERMrm VR128:$src1, VR128:$src2, addr:$src3)>;
def : Pat<(int_x86_xop_vpperm VR128:$src1, (bitconvert (memopv2i64 addr:$src2)),
                                  VR128:$src3),
          (VPPERMmr VR128:$src1, addr:$src2, VR128:$src3)>;

// VPCMOV
def : Pat<(int_x86_xop_vpcmov VR128:$src1, VR128:$src2, VR128:$src3),
          (VPCMOVrr VR128:$src1, VR128:$src2, VR128:$src3)>;
def : Pat<(int_x86_xop_vpcmov VR128:$src1, VR128:$src2,
                                  (bitconvert (memopv2i64 addr:$src3))),
          (VPCMOVrm VR128:$src1, VR128:$src2, addr:$src3)>;
def : Pat<(int_x86_xop_vpcmov VR128:$src1, (bitconvert (memopv2i64 addr:$src2)),
                                  VR128:$src3),
          (VPCMOVmr VR128:$src1, addr:$src2, VR128:$src3)>;
def : Pat<(int_x86_xop_vpcmov_256 VR256:$src1, VR256:$src2, VR256:$src3),
          (VPCMOVrrY VR256:$src1, VR256:$src2, VR256:$src3)>;
def : Pat<(int_x86_xop_vpcmov_256 VR256:$src1, VR256:$src2,
                                  (bitconvert (memopv4i64 addr:$src3))),
          (VPCMOVrmY VR256:$src1, VR256:$src2, addr:$src3)>;
def : Pat<(int_x86_xop_vpcmov_256 VR256:$src1,
                                  (bitconvert (memopv4i64 addr:$src2)),
                                  VR256:$src3),
          (VPCMOVmrY VR256:$src1, addr:$src2, VR256:$src3)>;

// VPCMOV di
def : Pat<(int_x86_xop_vpcmov_v2di VR128:$src1, VR128:$src2, VR128:$src3),
          (VPCMOVrr VR128:$src1, VR128:$src2, VR128:$src3)>;
def : Pat<(int_x86_xop_vpcmov_v2di VR128:$src1, VR128:$src2,
                                  (bitconvert (memopv2i64 addr:$src3))),
          (VPCMOVrm VR128:$src1, VR128:$src2, addr:$src3)>;
def : Pat<(int_x86_xop_vpcmov_v2di VR128:$src1,
                                   (bitconvert (memopv2i64 addr:$src2)),
                                   VR128:$src3),
          (VPCMOVmr VR128:$src1, addr:$src2, VR128:$src3)>;
def : Pat<(int_x86_xop_vpcmov_v4di_256 VR256:$src1, VR256:$src2, VR256:$src3),
          (VPCMOVrrY VR256:$src1, VR256:$src2, VR256:$src3)>;
def : Pat<(int_x86_xop_vpcmov_v4di_256 VR256:$src1, VR256:$src2,
                                  (bitconvert (memopv2i64 addr:$src3))),
          (VPCMOVrmY VR256:$src1, VR256:$src2, addr:$src3)>;
def : Pat<(int_x86_xop_vpcmov_v4di_256 VR256:$src1,
                                      (bitconvert (memopv2i64 addr:$src2)),
                                      VR256:$src3),
          (VPCMOVmrY VR256:$src1, addr:$src2, VR256:$src3)>;

// VPCMOV si
def : Pat<(int_x86_xop_vpcmov_v4si VR128:$src1, VR128:$src2, VR128:$src3),
          (VPCMOVrr VR128:$src1, VR128:$src2, VR128:$src3)>;
def : Pat<(int_x86_xop_vpcmov_v4si VR128:$src1, VR128:$src2,
                                  (bitconvert (memopv2i64 addr:$src3))),
          (VPCMOVrm VR128:$src1, VR128:$src2, addr:$src3)>;
def : Pat<(int_x86_xop_vpcmov_v4si VR128:$src1,
                                   (bitconvert (memopv2i64 addr:$src2)),
                                   VR128:$src3),
          (VPCMOVmr VR128:$src1, addr:$src2, VR128:$src3)>;
def : Pat<(int_x86_xop_vpcmov_v8si_256 VR256:$src1, VR256:$src2, VR256:$src3),
          (VPCMOVrrY VR256:$src1, VR256:$src2, VR256:$src3)>;
def : Pat<(int_x86_xop_vpcmov_v8si_256 VR256:$src1, VR256:$src2,
                                  (bitconvert (memopv2i64 addr:$src3))),
          (VPCMOVrmY VR256:$src1, VR256:$src2, addr:$src3)>;
def : Pat<(int_x86_xop_vpcmov_v8si_256 VR256:$src1,
                                      (bitconvert (memopv2i64 addr:$src2)),
                                      VR256:$src3),
          (VPCMOVmrY VR256:$src1, addr:$src2, VR256:$src3)>;


// VPCMOV hi
def : Pat<(int_x86_xop_vpcmov_v8hi VR128:$src1, VR128:$src2, VR128:$src3),
          (VPCMOVrr VR128:$src1, VR128:$src2, VR128:$src3)>;
def : Pat<(int_x86_xop_vpcmov_v8hi VR128:$src1, VR128:$src2,
                                  (bitconvert (memopv2i64 addr:$src3))),
          (VPCMOVrm VR128:$src1, VR128:$src2, addr:$src3)>;
def : Pat<(int_x86_xop_vpcmov_v8hi VR128:$src1,
                                   (bitconvert (memopv2i64 addr:$src2)),
                                   VR128:$src3),
          (VPCMOVmr VR128:$src1, addr:$src2, VR128:$src3)>;
def : Pat<(int_x86_xop_vpcmov_v16hi_256 VR256:$src1, VR256:$src2, VR256:$src3),
          (VPCMOVrrY VR256:$src1, VR256:$src2, VR256:$src3)>;
def : Pat<(int_x86_xop_vpcmov_v16hi_256 VR256:$src1, VR256:$src2,
                                  (bitconvert (memopv2i64 addr:$src3))),
          (VPCMOVrmY VR256:$src1, VR256:$src2, addr:$src3)>;
def : Pat<(int_x86_xop_vpcmov_v16hi_256 VR256:$src1,
                                      (bitconvert (memopv2i64 addr:$src2)),
                                      VR256:$src3),
          (VPCMOVmrY VR256:$src1, addr:$src2, VR256:$src3)>;

// VPCMOV qi
def : Pat<(int_x86_xop_vpcmov_v16qi VR128:$src1, VR128:$src2, VR128:$src3),
          (VPCMOVrr VR128:$src1, VR128:$src2, VR128:$src3)>;
def : Pat<(int_x86_xop_vpcmov_v16qi VR128:$src1, VR128:$src2,
                                  (bitconvert (memopv2i64 addr:$src3))),
          (VPCMOVrm VR128:$src1, VR128:$src2, addr:$src3)>;
def : Pat<(int_x86_xop_vpcmov_v16qi VR128:$src1,
                                    (bitconvert (memopv2i64 addr:$src2)),
                                    VR128:$src3),
          (VPCMOVmr VR128:$src1, addr:$src2, VR128:$src3)>;
def : Pat<(int_x86_xop_vpcmov_v32qi_256 VR256:$src1, VR256:$src2, VR256:$src3),
          (VPCMOVrrY VR256:$src1, VR256:$src2, VR256:$src3)>;
def : Pat<(int_x86_xop_vpcmov_v32qi_256 VR256:$src1, VR256:$src2,
                                  (bitconvert (memopv2i64 addr:$src3))),
          (VPCMOVrmY VR256:$src1, VR256:$src2, addr:$src3)>;
def : Pat<(int_x86_xop_vpcmov_v32qi_256 VR256:$src1,
                                      (bitconvert (memopv2i64 addr:$src2)),
                                      VR256:$src3),
          (VPCMOVmrY VR256:$src1, addr:$src2, VR256:$src3)>;

// VPCMOV df
def : Pat<(int_x86_xop_vpcmov_v2df VR128:$src1, VR128:$src2, VR128:$src3),
          (VPCMOVrr VR128:$src1, VR128:$src2, VR128:$src3)>;
def : Pat<(int_x86_xop_vpcmov_v2df VR128:$src1, VR128:$src2,
                                  (bitconvert (memopv2i64 addr:$src3))),
          (VPCMOVrm VR128:$src1, VR128:$src2, addr:$src3)>;
def : Pat<(int_x86_xop_vpcmov_v2df VR128:$src1,
                                   (bitconvert (memopv2i64 addr:$src2)),
                                   VR128:$src3),
          (VPCMOVmr VR128:$src1, addr:$src2, VR128:$src3)>;
def : Pat<(int_x86_xop_vpcmov_v4df_256 VR256:$src1, VR256:$src2, VR256:$src3),
          (VPCMOVrrY VR256:$src1, VR256:$src2, VR256:$src3)>;
def : Pat<(int_x86_xop_vpcmov_v4df_256 VR256:$src1, VR256:$src2,
                                  (bitconvert (memopv2i64 addr:$src3))),
          (VPCMOVrmY VR256:$src1, VR256:$src2, addr:$src3)>;
def : Pat<(int_x86_xop_vpcmov_v4df_256 VR256:$src1,
                                      (bitconvert (memopv2i64 addr:$src2)),
                                      VR256:$src3),
          (VPCMOVmrY VR256:$src1, addr:$src2, VR256:$src3)>;

// VPCMOV sf
def : Pat<(int_x86_xop_vpcmov_v4sf VR128:$src1, VR128:$src2, VR128:$src3),
          (VPCMOVrr VR128:$src1, VR128:$src2, VR128:$src3)>;
def : Pat<(int_x86_xop_vpcmov_v4sf VR128:$src1, VR128:$src2,
                                  (bitconvert (memopv2i64 addr:$src3))),
          (VPCMOVrm VR128:$src1, VR128:$src2, addr:$src3)>;
def : Pat<(int_x86_xop_vpcmov_v4sf VR128:$src1,
                                   (bitconvert (memopv2i64 addr:$src2)),
                                   VR128:$src3),
          (VPCMOVmr VR128:$src1, addr:$src2, VR128:$src3)>;
def : Pat<(int_x86_xop_vpcmov_v8sf_256 VR256:$src1, VR256:$src2, VR256:$src3),
          (VPCMOVrrY VR256:$src1, VR256:$src2, VR256:$src3)>;
def : Pat<(int_x86_xop_vpcmov_v8sf_256 VR256:$src1, VR256:$src2,
                                  (bitconvert (memopv2i64 addr:$src3))),
          (VPCMOVrmY VR256:$src1, VR256:$src2, addr:$src3)>;
def : Pat<(int_x86_xop_vpcmov_v8sf_256 VR256:$src1,
                                      (bitconvert (memopv2i64 addr:$src2)),
                                      VR256:$src3),
          (VPCMOVmrY VR256:$src1, addr:$src2, VR256:$src3)>;

