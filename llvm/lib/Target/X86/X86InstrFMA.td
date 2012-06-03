//===-- X86InstrFMA.td - FMA Instruction Set ---------------*- tablegen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file describes FMA (Fused Multiply-Add) instructions.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// FMA3 - Intel 3 operand Fused Multiply-Add instructions
//===----------------------------------------------------------------------===//

let Constraints = "$src1 = $dst" in {
multiclass fma3p_rm<bits<8> opc, string OpcodeStr> {
let neverHasSideEffects = 1 in {
  def r : FMA3<opc, MRMSrcReg, (outs VR128:$dst),
           (ins VR128:$src1, VR128:$src2, VR128:$src3),
           !strconcat(OpcodeStr, "\t{$src3, $src2, $dst|$dst, $src2, $src3}"),
           []>;
  let mayLoad = 1 in
  def m : FMA3<opc, MRMSrcMem, (outs VR128:$dst),
           (ins VR128:$src1, VR128:$src2, f128mem:$src3),
           !strconcat(OpcodeStr, "\t{$src3, $src2, $dst|$dst, $src2, $src3}"),
           []>;
  def rY : FMA3<opc, MRMSrcReg, (outs VR256:$dst),
           (ins VR256:$src1, VR256:$src2, VR256:$src3),
           !strconcat(OpcodeStr, "\t{$src3, $src2, $dst|$dst, $src2, $src3}"),
           []>;
  let mayLoad = 1 in
  def mY : FMA3<opc, MRMSrcMem, (outs VR256:$dst),
           (ins VR256:$src1, VR256:$src2, f256mem:$src3),
           !strconcat(OpcodeStr, "\t{$src3, $src2, $dst|$dst, $src2, $src3}"),
           []>;
} // neverHasSideEffects = 1
}

// Intrinsic for 132 pattern
multiclass fma3p_rm_int<bits<8> opc, string OpcodeStr,
                        PatFrag MemFrag128, PatFrag MemFrag256,
                        Intrinsic Int128, Intrinsic Int256> {
  def r_Int : FMA3<opc, MRMSrcReg, (outs VR128:$dst),
           (ins VR128:$src1, VR128:$src2, VR128:$src3),
           !strconcat(OpcodeStr, "\t{$src3, $src2, $dst|$dst, $src2, $src3}"),
           [(set VR128:$dst, (Int128 VR128:$src1, VR128:$src3, VR128:$src2))]>;
  def m_Int : FMA3<opc, MRMSrcMem, (outs VR128:$dst),
           (ins VR128:$src1, VR128:$src2, f128mem:$src3),
           !strconcat(OpcodeStr, "\t{$src3, $src2, $dst|$dst, $src2, $src3}"),
           [(set VR128:$dst,
             (Int128 VR128:$src1, (MemFrag128 addr:$src3), VR128:$src2))]>;
  def rY_Int : FMA3<opc, MRMSrcReg, (outs VR256:$dst),
           (ins VR256:$src1, VR256:$src2, VR256:$src3),
           !strconcat(OpcodeStr, "\t{$src3, $src2, $dst|$dst, $src2, $src3}"),
           [(set VR256:$dst, (Int256 VR256:$src1, VR256:$src3, VR256:$src2))]>;
  def mY_Int : FMA3<opc, MRMSrcMem, (outs VR256:$dst),
           (ins VR256:$src1, VR256:$src2, f256mem:$src3),
           !strconcat(OpcodeStr, "\t{$src3, $src2, $dst|$dst, $src2, $src3}"),
           [(set VR256:$dst,
             (Int256 VR256:$src1, (MemFrag256 addr:$src3), VR256:$src2))]>;
}
} // Constraints = "$src1 = $dst"

multiclass fma3p_forms<bits<8> opc132, bits<8> opc213, bits<8> opc231,
                       string OpcodeStr, string PackTy,
                       PatFrag MemFrag128, PatFrag MemFrag256,
                       Intrinsic Int128, Intrinsic Int256> {
  defm r132 : fma3p_rm_int <opc132, !strconcat(OpcodeStr,
                            !strconcat("132", PackTy)), MemFrag128, MemFrag256,
                            Int128, Int256>;
  defm r132 : fma3p_rm <opc132, !strconcat(OpcodeStr, !strconcat("132", PackTy))>;
  defm r213 : fma3p_rm <opc213, !strconcat(OpcodeStr, !strconcat("213", PackTy))>;
  defm r231 : fma3p_rm <opc231, !strconcat(OpcodeStr, !strconcat("231", PackTy))>;
}

// Fused Multiply-Add
let ExeDomain = SSEPackedSingle in {
  defm VFMADDPS    : fma3p_forms<0x98, 0xA8, 0xB8, "vfmadd", "ps", memopv4f32,
    memopv8f32, int_x86_fma_vfmadd_ps, int_x86_fma_vfmadd_ps_256>;
  defm VFMSUBPS    : fma3p_forms<0x9A, 0xAA, 0xBA, "vfmsub", "ps",  memopv4f32,
    memopv8f32, int_x86_fma_vfmsub_ps, int_x86_fma_vfmsub_ps_256>;
  defm VFMADDSUBPS : fma3p_forms<0x96, 0xA6, 0xB6, "vfmaddsub", "ps",
    memopv4f32, memopv8f32, int_x86_fma_vfmaddsub_ps,
    int_x86_fma_vfmaddsub_ps_256>;
  defm VFMSUBADDPS : fma3p_forms<0x97, 0xA7, 0xB7, "vfmsubadd", "ps",
    memopv4f32, memopv8f32, int_x86_fma_vfmsubadd_ps,
    int_x86_fma_vfmaddsub_ps_256>;
}

let ExeDomain = SSEPackedDouble in {
  defm VFMADDPD    : fma3p_forms<0x98, 0xA8, 0xB8, "vfmadd", "pd", memopv2f64,
    memopv4f64, int_x86_fma_vfmadd_pd, int_x86_fma_vfmadd_pd_256>, VEX_W;
  defm VFMSUBPD    : fma3p_forms<0x9A, 0xAA, 0xBA, "vfmsub", "pd", memopv2f64,
    memopv4f64, int_x86_fma_vfmsub_pd, int_x86_fma_vfmsub_pd_256>, VEX_W;
  defm VFMADDSUBPD : fma3p_forms<0x96, 0xA6, 0xB6, "vfmaddsub", "pd", memopv2f64,
    memopv4f64, int_x86_fma_vfmaddsub_pd, int_x86_fma_vfmaddsub_pd_256>, VEX_W;
  defm VFMSUBADDPD : fma3p_forms<0x97, 0xA7, 0xB7, "vfmsubadd", "pd", memopv2f64,
    memopv4f64, int_x86_fma_vfmsubadd_pd, int_x86_fma_vfmsubadd_pd_256>, VEX_W;
}

// Fused Negative Multiply-Add
let ExeDomain = SSEPackedSingle in {
  defm VFNMADDPS : fma3p_forms<0x9C, 0xAC, 0xBC, "vfnmadd", "ps",  memopv4f32,
    memopv8f32, int_x86_fma_vfnmadd_ps, int_x86_fma_vfnmadd_ps_256>;
  defm VFNMSUBPS : fma3p_forms<0x9E, 0xAE, 0xBE, "vfnmsub", "ps",  memopv4f32,
    memopv8f32, int_x86_fma_vfnmsub_ps, int_x86_fma_vfnmsub_ps_256>;
}
let ExeDomain = SSEPackedDouble in {
  defm VFNMADDPD : fma3p_forms<0x9C, 0xAC, 0xBC, "vfnmadd", "pd", memopv2f64,
    memopv4f64, int_x86_fma_vfnmadd_pd, int_x86_fma_vfnmadd_pd_256>, VEX_W;
  defm VFNMSUBPD : fma3p_forms<0x9E, 0xAE, 0xBE, "vfnmsub", "pd", memopv2f64,
    memopv4f64, int_x86_fma_vfnmsub_pd, int_x86_fma_vfnmsub_pd_256>, VEX_W;
}


let Constraints = "$src1 = $dst" in {
multiclass fma3s_rm<bits<8> opc, string OpcodeStr, X86MemOperand x86memop,
                    RegisterClass RC> {
let neverHasSideEffects = 1 in {
  def r : FMA3<opc, MRMSrcReg, (outs RC:$dst),
           (ins RC:$src1, RC:$src2, RC:$src3),
           !strconcat(OpcodeStr, "\t{$src3, $src2, $dst|$dst, $src2, $src3}"),
           []>;
  let mayLoad = 1 in
  def m : FMA3<opc, MRMSrcMem, (outs RC:$dst),
           (ins RC:$src1, RC:$src2, x86memop:$src3),
           !strconcat(OpcodeStr, "\t{$src3, $src2, $dst|$dst, $src2, $src3}"),
           []>;
} // neverHasSideEffects = 1
}

multiclass fma3s_rm_int<bits<8> opc, string OpcodeStr, Operand memop,
                        ComplexPattern mem_cpat, Intrinsic IntId> {
  def r_Int : FMA3<opc, MRMSrcReg, (outs VR128:$dst),
           (ins VR128:$src1, VR128:$src2, VR128:$src3),
           !strconcat(OpcodeStr, "\t{$src3, $src2, $dst|$dst, $src2, $src3}"),
           [(set VR128:$dst, (IntId VR128:$src1, VR128:$src3, VR128:$src2))]>;
  def m_Int : FMA3<opc, MRMSrcMem, (outs VR128:$dst),
           (ins VR128:$src1, VR128:$src2, memop:$src3),
           !strconcat(OpcodeStr, "\t{$src3, $src2, $dst|$dst, $src2, $src3}"),
           [(set VR128:$dst,
             (IntId VR128:$src1, mem_cpat:$src3, VR128:$src2))]>;
}
} // Constraints = "$src1 = $dst"

multiclass fma3s_forms<bits<8> opc132, bits<8> opc213, bits<8> opc231,
                       string OpStr, Intrinsic IntF32, Intrinsic IntF64> {
  defm SSr132 : fma3s_rm<opc132, !strconcat(OpStr, "132ss"), f32mem, FR32>;
  defm SSr213 : fma3s_rm<opc213, !strconcat(OpStr, "213ss"), f32mem, FR32>;
  defm SSr231 : fma3s_rm<opc231, !strconcat(OpStr, "231ss"), f32mem, FR32>;
  defm SDr132 : fma3s_rm<opc132, !strconcat(OpStr, "132sd"), f64mem, FR64>, VEX_W;
  defm SDr213 : fma3s_rm<opc213, !strconcat(OpStr, "213sd"), f64mem, FR64>, VEX_W;
  defm SDr231 : fma3s_rm<opc231, !strconcat(OpStr, "231sd"), f64mem, FR64>, VEX_W;
  defm SSr132_Int : fma3s_rm_int <opc132, !strconcat(OpStr, "132ss"), ssmem,
                                  sse_load_f32, IntF32>;
  defm SDr132_Int : fma3s_rm_int <opc132, !strconcat(OpStr, "132sd"), sdmem,
                                  sse_load_f64, IntF64>;
}

defm VFMADD : fma3s_forms<0x99, 0xA9, 0xB9, "vfmadd", int_x86_fma_vfmadd_ss,
                          int_x86_fma_vfmadd_sd>, VEX_LIG;
defm VFMSUB : fma3s_forms<0x9B, 0xAB, 0xBB, "vfmsub", int_x86_fma_vfmsub_ss,
                          int_x86_fma_vfmsub_sd>, VEX_LIG;

defm VFNMADD : fma3s_forms<0x9D, 0xAD, 0xBD, "vfnmadd", int_x86_fma_vfnmadd_ss,
                           int_x86_fma_vfnmadd_sd>, VEX_LIG;
defm VFNMSUB : fma3s_forms<0x9F, 0xAF, 0xBF, "vfnmsub", int_x86_fma_vfnmsub_ss,
                           int_x86_fma_vfnmsub_sd>, VEX_LIG;


//===----------------------------------------------------------------------===//
// FMA4 - AMD 4 operand Fused Multiply-Add instructions
//===----------------------------------------------------------------------===//


multiclass fma4s<bits<8> opc, string OpcodeStr, Operand memop,
                 ComplexPattern mem_cpat, Intrinsic Int> {
  def rr : FMA4<opc, MRMSrcReg, (outs VR128:$dst),
           (ins VR128:$src1, VR128:$src2, VR128:$src3),
           !strconcat(OpcodeStr,
           "\t{$src3, $src2, $src1, $dst|$dst, $src1, $src2, $src3}"),
           [(set VR128:$dst,
             (Int VR128:$src1, VR128:$src2, VR128:$src3))]>, VEX_W, MemOp4;
  def rm : FMA4<opc, MRMSrcMem, (outs VR128:$dst),
           (ins VR128:$src1, VR128:$src2, memop:$src3),
           !strconcat(OpcodeStr,
           "\t{$src3, $src2, $src1, $dst|$dst, $src1, $src2, $src3}"),
           [(set VR128:$dst,
             (Int VR128:$src1, VR128:$src2, mem_cpat:$src3))]>, VEX_W, MemOp4;
  def mr : FMA4<opc, MRMSrcMem, (outs VR128:$dst),
           (ins VR128:$src1, memop:$src2, VR128:$src3),
           !strconcat(OpcodeStr,
           "\t{$src3, $src2, $src1, $dst|$dst, $src1, $src2, $src3}"),
           [(set VR128:$dst,
             (Int VR128:$src1, mem_cpat:$src2, VR128:$src3))]>;
// For disassembler
let isCodeGenOnly = 1 in
  def rr_REV : FMA4<opc, MRMSrcReg, (outs VR128:$dst),
               (ins VR128:$src1, VR128:$src2, VR128:$src3),
               !strconcat(OpcodeStr,
               "\t{$src3, $src2, $src1, $dst|$dst, $src1, $src2, $src3}"), []>;
}

multiclass fma4p<bits<8> opc, string OpcodeStr,
                 Intrinsic Int128, Intrinsic Int256,
                 PatFrag ld_frag128, PatFrag ld_frag256> {
  def rr : FMA4<opc, MRMSrcReg, (outs VR128:$dst),
           (ins VR128:$src1, VR128:$src2, VR128:$src3),
           !strconcat(OpcodeStr,
           "\t{$src3, $src2, $src1, $dst|$dst, $src1, $src2, $src3}"),
           [(set VR128:$dst,
             (Int128 VR128:$src1, VR128:$src2, VR128:$src3))]>, VEX_W, MemOp4;
  def rm : FMA4<opc, MRMSrcMem, (outs VR128:$dst),
           (ins VR128:$src1, VR128:$src2, f128mem:$src3),
           !strconcat(OpcodeStr,
           "\t{$src3, $src2, $src1, $dst|$dst, $src1, $src2, $src3}"),
           [(set VR128:$dst, (Int128 VR128:$src1, VR128:$src2,
                              (ld_frag128 addr:$src3)))]>, VEX_W, MemOp4;
  def mr : FMA4<opc, MRMSrcMem, (outs VR128:$dst),
           (ins VR128:$src1, f128mem:$src2, VR128:$src3),
           !strconcat(OpcodeStr,
           "\t{$src3, $src2, $src1, $dst|$dst, $src1, $src2, $src3}"),
           [(set VR128:$dst,
             (Int128 VR128:$src1, (ld_frag128 addr:$src2), VR128:$src3))]>;
  def rrY : FMA4<opc, MRMSrcReg, (outs VR256:$dst),
           (ins VR256:$src1, VR256:$src2, VR256:$src3),
           !strconcat(OpcodeStr,
           "\t{$src3, $src2, $src1, $dst|$dst, $src1, $src2, $src3}"),
           [(set VR256:$dst,
             (Int256 VR256:$src1, VR256:$src2, VR256:$src3))]>, VEX_W, MemOp4;
  def rmY : FMA4<opc, MRMSrcMem, (outs VR256:$dst),
           (ins VR256:$src1, VR256:$src2, f256mem:$src3),
           !strconcat(OpcodeStr,
           "\t{$src3, $src2, $src1, $dst|$dst, $src1, $src2, $src3}"),
           [(set VR256:$dst, (Int256 VR256:$src1, VR256:$src2,
                              (ld_frag256 addr:$src3)))]>, VEX_W, MemOp4;
  def mrY : FMA4<opc, MRMSrcMem, (outs VR256:$dst),
           (ins VR256:$src1, f256mem:$src2, VR256:$src3),
           !strconcat(OpcodeStr,
           "\t{$src3, $src2, $src1, $dst|$dst, $src1, $src2, $src3}"),
           [(set VR256:$dst,
             (Int256 VR256:$src1, (ld_frag256 addr:$src2), VR256:$src3))]>;
// For disassembler
let isCodeGenOnly = 1 in {
  def rr_REV : FMA4<opc, MRMSrcReg, (outs VR128:$dst),
               (ins VR128:$src1, VR128:$src2, VR128:$src3),
               !strconcat(OpcodeStr,
               "\t{$src3, $src2, $src1, $dst|$dst, $src1, $src2, $src3}"), []>;
  def rrY_REV : FMA4<opc, MRMSrcReg, (outs VR256:$dst),
                (ins VR256:$src1, VR256:$src2, VR256:$src3),
                !strconcat(OpcodeStr,
                "\t{$src3, $src2, $src1, $dst|$dst, $src1, $src2, $src3}"), []>;
} // isCodeGenOnly = 1
}

let Predicates = [HasFMA4] in {

defm VFMADDSS4    : fma4s<0x6A, "vfmaddss", ssmem, sse_load_f32,
                          int_x86_fma_vfmadd_ss>;
defm VFMADDSD4    : fma4s<0x6B, "vfmaddsd", sdmem, sse_load_f64,
                          int_x86_fma_vfmadd_sd>;
defm VFMADDPS4    : fma4p<0x68, "vfmaddps", int_x86_fma_vfmadd_ps,
                          int_x86_fma_vfmadd_ps_256, memopv4f32, memopv8f32>;
defm VFMADDPD4    : fma4p<0x69, "vfmaddpd", int_x86_fma_vfmadd_pd,
                          int_x86_fma_vfmadd_pd_256, memopv2f64, memopv4f64>;
defm VFMSUBSS4    : fma4s<0x6E, "vfmsubss", ssmem, sse_load_f32,
                          int_x86_fma_vfmsub_ss>;
defm VFMSUBSD4    : fma4s<0x6F, "vfmsubsd", sdmem, sse_load_f64,
                          int_x86_fma_vfmsub_sd>;
defm VFMSUBPS4    : fma4p<0x6C, "vfmsubps", int_x86_fma_vfmsub_ps,
                          int_x86_fma_vfmsub_ps_256, memopv4f32, memopv8f32>;
defm VFMSUBPD4    : fma4p<0x6D, "vfmsubpd", int_x86_fma_vfmsub_pd,
                          int_x86_fma_vfmsub_pd_256, memopv2f64, memopv4f64>;
defm VFNMADDSS4   : fma4s<0x7A, "vfnmaddss", ssmem, sse_load_f32,
                          int_x86_fma_vfnmadd_ss>;
defm VFNMADDSD4   : fma4s<0x7B, "vfnmaddsd", sdmem, sse_load_f64,
                          int_x86_fma_vfnmadd_sd>;
defm VFNMADDPS4   : fma4p<0x78, "vfnmaddps", int_x86_fma_vfnmadd_ps,
                          int_x86_fma_vfnmadd_ps_256, memopv4f32, memopv8f32>;
defm VFNMADDPD4   : fma4p<0x79, "vfnmaddpd", int_x86_fma_vfnmadd_pd,
                          int_x86_fma_vfnmadd_pd_256, memopv2f64, memopv4f64>;
defm VFNMSUBSS4   : fma4s<0x7E, "vfnmsubss", ssmem, sse_load_f32,
                          int_x86_fma_vfnmsub_ss>;
defm VFNMSUBSD4   : fma4s<0x7F, "vfnmsubsd", sdmem, sse_load_f64,
                          int_x86_fma_vfnmsub_sd>;
defm VFNMSUBPS4   : fma4p<0x7C, "vfnmsubps", int_x86_fma_vfnmsub_ps,
                          int_x86_fma_vfnmsub_ps_256, memopv4f32, memopv8f32>;
defm VFNMSUBPD4   : fma4p<0x7D, "vfnmsubpd", int_x86_fma_vfnmsub_pd,
                          int_x86_fma_vfnmsub_pd_256, memopv2f64, memopv4f64>;
defm VFMADDSUBPS4 : fma4p<0x5C, "vfmaddsubps", int_x86_fma_vfmaddsub_ps,
                          int_x86_fma_vfmaddsub_ps_256, memopv4f32, memopv8f32>;
defm VFMADDSUBPD4 : fma4p<0x5D, "vfmaddsubpd", int_x86_fma_vfmaddsub_pd,
                          int_x86_fma_vfmaddsub_pd_256, memopv2f64, memopv4f64>;
defm VFMSUBADDPS4 : fma4p<0x5E, "vfmsubaddps", int_x86_fma_vfmsubadd_ps,
                          int_x86_fma_vfmsubadd_ps_256, memopv4f32, memopv8f32>;
defm VFMSUBADDPD4 : fma4p<0x5F, "vfmsubaddpd", int_x86_fma_vfmsubadd_pd,
                          int_x86_fma_vfmsubadd_pd_256, memopv2f64, memopv4f64>;
} // HasFMA4

