//====- X86InstrXOP.td - Describe the X86 Instruction Set --*- tablegen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file describes XOP (eXtended OPerations)
//
//===----------------------------------------------------------------------===//

multiclass xop2op<bits<8> opc, string OpcodeStr, X86MemOperand x86memop> {
  def rr : IXOP<opc, MRMSrcReg, (outs VR128:$dst), (ins VR128:$src),
           !strconcat(OpcodeStr, "\t{$src, $dst|$dst, $src}"),
           []>, VEX;
  def rm : IXOP<opc, MRMSrcMem, (outs VR128:$dst), (ins x86memop:$src),
           !strconcat(OpcodeStr, "\t{$src, $dst|$dst, $src}"),
           []>, VEX;
}

let isAsmParserOnly = 1 in {
  defm VPHSUBWD  : xop2op<0xE2, "vphsubwd", f128mem>;
  defm VPHSUBDQ  : xop2op<0xE3, "vphsubdq", f128mem>;
  defm VPHSUBBW  : xop2op<0xE1, "vphsubbw", f128mem>;
  defm VPHADDWQ  : xop2op<0xC7, "vphaddwq", f128mem>;
  defm VPHADDWD  : xop2op<0xC6, "vphaddwd", f128mem>;
  defm VPHADDUWQ : xop2op<0xD7, "vphadduwq", f128mem>;
  defm VPHADDUWD : xop2op<0xD6, "vphadduwd", f128mem>;
  defm VPHADDUDQ : xop2op<0xDB, "vphaddudq", f128mem>;
  defm VPHADDUBW : xop2op<0xD1, "vphaddubw", f128mem>;
  defm VPHADDUBQ : xop2op<0xD3, "vphaddubq", f128mem>;
  defm VPHADDUBD : xop2op<0xD2, "vphaddubd", f128mem>;
  defm VPHADDDQ  : xop2op<0xCB, "vphadddq", f128mem>;
  defm VPHADDBW  : xop2op<0xC1, "vphaddbw", f128mem>;
  defm VPHADDBQ  : xop2op<0xC3, "vphaddbq", f128mem>;
  defm VPHADDBD  : xop2op<0xC2, "vphaddbd", f128mem>;
  defm VFRCZSS   : xop2op<0x82, "vfrczss", f32mem>;
  defm VFRCZSD   : xop2op<0x83, "vfrczsd", f64mem>;
  defm VFRCZPS   : xop2op<0x80, "vfrczps", f128mem>;
  defm VFRCZPD   : xop2op<0x81, "vfrczpd", f128mem>;
}

multiclass xop2op256<bits<8> opc, string OpcodeStr> {
  def rrY : IXOP<opc, MRMSrcReg, (outs VR256:$dst), (ins VR256:$src),
           !strconcat(OpcodeStr, "\t{$src, $dst|$dst, $src}"),
           []>, VEX, VEX_L;
  def rmY : IXOP<opc, MRMSrcMem, (outs VR256:$dst), (ins f256mem:$src),
           !strconcat(OpcodeStr, "\t{$src, $dst|$dst, $src}"),
           []>, VEX;
}

let isAsmParserOnly = 1 in {
  defm VFRCZPS : xop2op256<0x80, "vfrczps">;
  defm VFRCZPD : xop2op256<0x81, "vfrczpd">;
}

multiclass xop3op<bits<8> opc, string OpcodeStr> {
  def rr : IXOP<opc, MRMSrcReg, (outs VR128:$dst),
           (ins VR128:$src1, VR128:$src2),
           !strconcat(OpcodeStr, "\t{$src2, $src1, $dst|$dst, $src1, $src2}"),
           []>, VEX_4VOp3;
  def rm : IXOP<opc, MRMSrcMem, (outs VR128:$dst),
           (ins VR128:$src1, f128mem:$src2),
           !strconcat(OpcodeStr, "\t{$src2, $src1, $dst|$dst, $src1, $src2}"),
           []>, VEX_4V, VEX_W;
  def mr : IXOP<opc, MRMSrcMem, (outs VR128:$dst),
           (ins f128mem:$src1, VR128:$src2),
           !strconcat(OpcodeStr, "\t{$src2, $src1, $dst|$dst, $src1, $src2}"),
           []>, VEX_4VOp3;
}

let isAsmParserOnly = 1 in {
  defm VPSHLW : xop3op<0x95, "vpshlw">;
  defm VPSHLQ : xop3op<0x97, "vpshlq">;
  defm VPSHLD : xop3op<0x96, "vpshld">;
  defm VPSHLB : xop3op<0x94, "vpshlb">;
  defm VPSHAW : xop3op<0x99, "vpshaw">;
  defm VPSHAQ : xop3op<0x9B, "vpshaq">;
  defm VPSHAD : xop3op<0x9A, "vpshad">;
  defm VPSHAB : xop3op<0x98, "vpshab">;
  defm VPROTW : xop3op<0x91, "vprotw">;
  defm VPROTQ : xop3op<0x93, "vprotq">;
  defm VPROTD : xop3op<0x92, "vprotd">;
  defm VPROTB : xop3op<0x90, "vprotb">;
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
multiclass xop4opm2<bits<8> opc, string OpcodeStr> {
  def rr : IXOPi8<opc, MRMSrcReg, (outs VR128:$dst),
           (ins VR128:$src1, VR128:$src2, VR128:$src3),
           !strconcat(OpcodeStr,
           "\t{$src3, $src2, $src1, $dst|$dst, $src1, $src2, $src3}"),
           []>, VEX_4V, VEX_I8IMM;
  def rm : IXOPi8<opc, MRMSrcMem, (outs VR128:$dst),
           (ins VR128:$src1, f128mem:$src2, VR128:$src3),
           !strconcat(OpcodeStr,
           "\t{$src3, $src2, $src1, $dst|$dst, $src1, $src2, $src3}"),
           []>, VEX_4V, VEX_I8IMM;
}

let isAsmParserOnly = 1 in {
  defm VPMADCSWD  : xop4opm2<0xB6, "vpmadcswd">;
  defm VPMADCSSWD : xop4opm2<0xA6, "vpmadcsswd">;
  defm VPMACSWW   : xop4opm2<0x95, "vpmacsww">;
  defm VPMACSWD   : xop4opm2<0x96, "vpmacswd">;
  defm VPMACSSWW  : xop4opm2<0x85, "vpmacssww">;
  defm VPMACSSWD  : xop4opm2<0x86, "vpmacsswd">;
  defm VPMACSSDQL : xop4opm2<0x87, "vpmacssdql">;
  defm VPMACSSDQH : xop4opm2<0x8F, "vpmacssdqh">;
  defm VPMACSSDD  : xop4opm2<0x8E, "vpmacssdd">;
  defm VPMACSDQL  : xop4opm2<0x97, "vpmacsdql">;
  defm VPMACSDQH  : xop4opm2<0x9F, "vpmacsdqh">;
  defm VPMACSDD   : xop4opm2<0x9E, "vpmacsdd">;
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
           []>, VEX_4V, VEX_I8IMM, XOP_W;
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
           []>, VEX_4V, VEX_I8IMM, XOP_W;
  def mrY : IXOPi8<opc, MRMSrcMem, (outs VR256:$dst),
           (ins VR256:$src1, f256mem:$src2, VR256:$src3),
           !strconcat(OpcodeStr,
           "\t{$src3, $src2, $src1, $dst|$dst, $src1, $src2, $src3}"),
           []>, VEX_4V, VEX_I8IMM;
}

let isAsmParserOnly = 1 in {
  defm VPCMOV : xop4op256<0xA2, "vpcmov">;
}

multiclass xop5op<bits<8> opc, string OpcodeStr> {
  def rr : IXOP5<opc, MRMSrcReg, (outs VR128:$dst),
        (ins VR128:$src1, VR128:$src2, VR128:$src3, i8imm:$src4),
        !strconcat(OpcodeStr,
        "\t{$src4, $src3, $src2, $src1, $dst|$dst, $src1, $src2, $src3, $src4}"),
        []>;
  def rm : IXOP5<opc, MRMSrcMem, (outs VR128:$dst),
        (ins VR128:$src1, VR128:$src2, f128mem:$src3, i8imm:$src4),
        !strconcat(OpcodeStr,
        "\t{$src4, $src3, $src2, $src1, $dst|$dst, $src1, $src2, $src3, $src4}"),
        []>, XOP_W;
  def mr : IXOP5<opc, MRMSrcMem, (outs VR128:$dst),
        (ins VR128:$src1, f128mem:$src2, VR128:$src3, i8imm:$src4),
        !strconcat(OpcodeStr,
        "\t{$src4, $src3, $src2, $src1, $dst|$dst, $src1, $src2, $src3, $src4}"),
        []>;
  def rrY : IXOP5<opc, MRMSrcReg, (outs VR256:$dst),
        (ins VR256:$src1, VR256:$src2, VR256:$src3, i8imm:$src4),
        !strconcat(OpcodeStr,
        "\t{$src4, $src3, $src2, $src1, $dst|$dst, $src1, $src2, $src3, $src4}"),
        []>;
  def rmY : IXOP5<opc, MRMSrcMem, (outs VR256:$dst),
        (ins VR256:$src1, VR256:$src2, f256mem:$src3, i8imm:$src4),
        !strconcat(OpcodeStr,
        "\t{$src4, $src3, $src2, $src1, $dst|$dst, $src1, $src2, $src3, $src4}"),
        []>, XOP_W;
  def mrY : IXOP5<opc, MRMSrcMem, (outs VR256:$dst),
        (ins VR256:$src1, f256mem:$src2, VR256:$src3, i8imm:$src4),
        !strconcat(OpcodeStr,
        "\t{$src4, $src3, $src2, $src1, $dst|$dst, $src1, $src2, $src3, $src4}"),
        []>;
}

let isAsmParserOnly = 1 in {
  defm VPERMIL2PD : xop5op<0x49, "vpermil2pd">;
  defm VPERMIL2PS : xop5op<0x48, "vpermil2ps">;
}
