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
multiclass fma3p_rm<bits<8> opc, string OpcodeStr,
                    PatFrag MemFrag128, PatFrag MemFrag256,
                    ValueType OpVT128, ValueType OpVT256,
                    SDPatternOperator Op = null_frag, bit MayLoad = 1> {
  def r     : FMA3<opc, MRMSrcReg, (outs VR128:$dst),
                   (ins VR128:$src1, VR128:$src2, VR128:$src3),
                   !strconcat(OpcodeStr,
                              "\t{$src3, $src2, $dst|$dst, $src2, $src3}"),
                   [(set VR128:$dst, (OpVT128 (Op VR128:$src2,
                                               VR128:$src1, VR128:$src3)))]>;

  let mayLoad = MayLoad in
  def m     : FMA3<opc, MRMSrcMem, (outs VR128:$dst),
                   (ins VR128:$src1, VR128:$src2, f128mem:$src3),
                   !strconcat(OpcodeStr,
                              "\t{$src3, $src2, $dst|$dst, $src2, $src3}"),
                   [(set VR128:$dst, (OpVT128 (Op VR128:$src2, VR128:$src1,
                                               (MemFrag128 addr:$src3))))]>;

  def rY    : FMA3<opc, MRMSrcReg, (outs VR256:$dst),
                   (ins VR256:$src1, VR256:$src2, VR256:$src3),
                   !strconcat(OpcodeStr,
                              "\t{$src3, $src2, $dst|$dst, $src2, $src3}"),
                   [(set VR256:$dst, (OpVT256 (Op VR256:$src2, VR256:$src1,
                                               VR256:$src3)))]>;

  let mayLoad = MayLoad in
  def mY    : FMA3<opc, MRMSrcMem, (outs VR256:$dst),
                   (ins VR256:$src1, VR256:$src2, f256mem:$src3),
                   !strconcat(OpcodeStr,
                              "\t{$src3, $src2, $dst|$dst, $src2, $src3}"),
                   [(set VR256:$dst,
                     (OpVT256 (Op VR256:$src2, VR256:$src1,
                               (MemFrag256 addr:$src3))))]>;
}
} // Constraints = "$src1 = $dst"

multiclass fma3p_forms<bits<8> opc132, bits<8> opc213, bits<8> opc231,
                       string OpcodeStr, string PackTy,
                       PatFrag MemFrag128, PatFrag MemFrag256,
                       SDNode Op, ValueType OpTy128, ValueType OpTy256> {
  defm r213 : fma3p_rm<opc213,
                       !strconcat(OpcodeStr, !strconcat("213", PackTy)),
                       MemFrag128, MemFrag256, OpTy128, OpTy256, Op, 0>;
let neverHasSideEffects = 1 in {
  defm r132 : fma3p_rm<opc132,
                       !strconcat(OpcodeStr, !strconcat("132", PackTy)),
                       MemFrag128, MemFrag256, OpTy128, OpTy256>;
  defm r231 : fma3p_rm<opc231,
                       !strconcat(OpcodeStr, !strconcat("231", PackTy)),
                       MemFrag128, MemFrag256, OpTy128, OpTy256>;
} // neverHasSideEffects = 1
}

// Fused Multiply-Add
let ExeDomain = SSEPackedSingle in {
  defm VFMADDPS    : fma3p_forms<0x98, 0xA8, 0xB8, "vfmadd", "ps", memopv4f32,
                                 memopv8f32, X86Fmadd, v4f32, v8f32>;
  defm VFMSUBPS    : fma3p_forms<0x9A, 0xAA, 0xBA, "vfmsub", "ps", memopv4f32,
                                 memopv8f32, X86Fmsub, v4f32, v8f32>;
  defm VFMADDSUBPS : fma3p_forms<0x96, 0xA6, 0xB6, "vfmaddsub", "ps",
                                 memopv4f32, memopv8f32, X86Fmaddsub,
                                 v4f32, v8f32>;
  defm VFMSUBADDPS : fma3p_forms<0x97, 0xA7, 0xB7, "vfmsubadd", "ps",
                                 memopv4f32, memopv8f32, X86Fmsubadd,
                                 v4f32, v8f32>;
}

let ExeDomain = SSEPackedDouble in {
  defm VFMADDPD    : fma3p_forms<0x98, 0xA8, 0xB8, "vfmadd", "pd", memopv2f64,
                                 memopv4f64, X86Fmadd, v2f64, v4f64>, VEX_W;
  defm VFMSUBPD    : fma3p_forms<0x9A, 0xAA, 0xBA, "vfmsub", "pd", memopv2f64,
                                 memopv4f64, X86Fmsub, v2f64, v4f64>, VEX_W;
  defm VFMADDSUBPD : fma3p_forms<0x96, 0xA6, 0xB6, "vfmaddsub", "pd",
                                 memopv2f64, memopv4f64, X86Fmaddsub,
                                 v2f64, v4f64>, VEX_W;
  defm VFMSUBADDPD : fma3p_forms<0x97, 0xA7, 0xB7, "vfmsubadd", "pd",
                                 memopv2f64, memopv4f64, X86Fmsubadd,
                                 v2f64, v4f64>, VEX_W;
}

// Fused Negative Multiply-Add
let ExeDomain = SSEPackedSingle in {
  defm VFNMADDPS : fma3p_forms<0x9C, 0xAC, 0xBC, "vfnmadd", "ps",  memopv4f32,
                               memopv8f32, X86Fnmadd, v4f32, v8f32>;
  defm VFNMSUBPS : fma3p_forms<0x9E, 0xAE, 0xBE, "vfnmsub", "ps",  memopv4f32,
                               memopv8f32, X86Fnmsub, v4f32, v8f32>;
}
let ExeDomain = SSEPackedDouble in {
  defm VFNMADDPD : fma3p_forms<0x9C, 0xAC, 0xBC, "vfnmadd", "pd", memopv2f64,
                               memopv4f64, X86Fnmadd, v2f64, v4f64>, VEX_W;
  defm VFNMSUBPD : fma3p_forms<0x9E, 0xAE, 0xBE, "vfnmsub", "pd",
                               memopv2f64, memopv4f64, X86Fnmsub, v2f64,
                               v4f64>, VEX_W;
}

let Predicates = [HasFMA] in {
  def : Pat<(int_x86_fma_vfmadd_ps VR128:$src2, VR128:$src1, VR128:$src3),
            (VFMADDPSr213r VR128:$src1, VR128:$src2, VR128:$src3)>;
  def : Pat<(int_x86_fma_vfmadd_ps VR128:$src2, VR128:$src1,
             (memopv4f32 addr:$src3)),
            (VFMADDPSr213m VR128:$src1, VR128:$src2, addr:$src3)>;
  def : Pat<(int_x86_fma_vfmsub_ps VR128:$src2, VR128:$src1, VR128:$src3),
            (VFMSUBPSr213r VR128:$src1, VR128:$src2, VR128:$src3)>;
  def : Pat<(int_x86_fma_vfmsub_ps VR128:$src2, VR128:$src1,
             (memopv4f32 addr:$src3)),
            (VFMSUBPSr213m VR128:$src1, VR128:$src2, addr:$src3)>;
  def : Pat<(int_x86_fma_vfmaddsub_ps VR128:$src2, VR128:$src1, VR128:$src3),
            (VFMADDSUBPSr213r VR128:$src1, VR128:$src2, VR128:$src3)>;
  def : Pat<(int_x86_fma_vfmaddsub_ps VR128:$src2, VR128:$src1,
             (memopv4f32 addr:$src3)),
            (VFMADDSUBPSr213m VR128:$src1, VR128:$src2, addr:$src3)>;
  def : Pat<(int_x86_fma_vfmsubadd_ps VR128:$src2, VR128:$src1, VR128:$src3),
            (VFMSUBADDPSr213r VR128:$src1, VR128:$src2, VR128:$src3)>;
  def : Pat<(int_x86_fma_vfmsubadd_ps VR128:$src2, VR128:$src1,
             (memopv4f32 addr:$src3)),
            (VFMSUBADDPSr213m VR128:$src1, VR128:$src2, addr:$src3)>;

  def : Pat<(int_x86_fma_vfmadd_ps_256 VR256:$src2, VR256:$src1, VR256:$src3),
            (VFMADDPSr213rY VR256:$src1, VR256:$src2, VR256:$src3)>;
  def : Pat<(int_x86_fma_vfmadd_ps_256 VR256:$src2, VR256:$src1,
             (memopv8f32 addr:$src3)),
            (VFMADDPSr213mY VR256:$src1, VR256:$src2, addr:$src3)>;
  def : Pat<(int_x86_fma_vfmsub_ps_256 VR256:$src2, VR256:$src1, VR256:$src3),
            (VFMSUBPSr213rY VR256:$src1, VR256:$src2, VR256:$src3)>;
  def : Pat<(int_x86_fma_vfmsub_ps_256 VR256:$src2, VR256:$src1,
             (memopv8f32 addr:$src3)),
            (VFMSUBPSr213mY VR256:$src1, VR256:$src2, addr:$src3)>;
  def : Pat<(int_x86_fma_vfmaddsub_ps_256 VR256:$src2, VR256:$src1, VR256:$src3),
            (VFMADDSUBPSr213rY VR256:$src1, VR256:$src2, VR256:$src3)>;
  def : Pat<(int_x86_fma_vfmaddsub_ps_256 VR256:$src2, VR256:$src1,
             (memopv8f32 addr:$src3)),
            (VFMADDSUBPSr213mY VR256:$src1, VR256:$src2, addr:$src3)>;
  def : Pat<(int_x86_fma_vfmsubadd_ps_256 VR256:$src2, VR256:$src1, VR256:$src3),
            (VFMSUBADDPSr213rY VR256:$src1, VR256:$src2, VR256:$src3)>;
  def : Pat<(int_x86_fma_vfmsubadd_ps_256 VR256:$src2, VR256:$src1,
             (memopv8f32 addr:$src3)),
            (VFMSUBADDPSr213mY VR256:$src1, VR256:$src2, addr:$src3)>;

  def : Pat<(int_x86_fma_vfmadd_pd VR128:$src2, VR128:$src1, VR128:$src3),
            (VFMADDPDr213r VR128:$src1, VR128:$src2, VR128:$src3)>;
  def : Pat<(int_x86_fma_vfmadd_pd VR128:$src2, VR128:$src1,
             (memopv2f64 addr:$src3)),
            (VFMADDPDr213m VR128:$src1, VR128:$src2, addr:$src3)>;
  def : Pat<(int_x86_fma_vfmsub_pd VR128:$src2, VR128:$src1, VR128:$src3),
            (VFMSUBPDr213r VR128:$src1, VR128:$src2, VR128:$src3)>;
  def : Pat<(int_x86_fma_vfmsub_pd VR128:$src2, VR128:$src1,
             (memopv2f64 addr:$src3)),
            (VFMSUBPDr213m VR128:$src1, VR128:$src2, addr:$src3)>;
  def : Pat<(int_x86_fma_vfmaddsub_pd VR128:$src2, VR128:$src1, VR128:$src3),
            (VFMADDSUBPDr213r VR128:$src1, VR128:$src2, VR128:$src3)>;
  def : Pat<(int_x86_fma_vfmaddsub_pd VR128:$src2, VR128:$src1,
             (memopv2f64 addr:$src3)),
            (VFMADDSUBPDr213m VR128:$src1, VR128:$src2, addr:$src3)>;
  def : Pat<(int_x86_fma_vfmsubadd_pd VR128:$src2, VR128:$src1, VR128:$src3),
            (VFMSUBADDPDr213r VR128:$src1, VR128:$src2, VR128:$src3)>;
  def : Pat<(int_x86_fma_vfmsubadd_pd VR128:$src2, VR128:$src1,
             (memopv2f64 addr:$src3)),
            (VFMSUBADDPDr213m VR128:$src1, VR128:$src2, addr:$src3)>;

  def : Pat<(int_x86_fma_vfmadd_pd_256 VR256:$src2, VR256:$src1, VR256:$src3),
            (VFMADDPDr213rY VR256:$src1, VR256:$src2, VR256:$src3)>;
  def : Pat<(int_x86_fma_vfmadd_pd_256 VR256:$src2, VR256:$src1,
             (memopv4f64 addr:$src3)),
            (VFMADDPDr213mY VR256:$src1, VR256:$src2, addr:$src3)>;
  def : Pat<(int_x86_fma_vfmsub_pd_256 VR256:$src2, VR256:$src1, VR256:$src3),
            (VFMSUBPDr213rY VR256:$src1, VR256:$src2, VR256:$src3)>;
  def : Pat<(int_x86_fma_vfmsub_pd_256 VR256:$src2, VR256:$src1,
             (memopv4f64 addr:$src3)),
            (VFMSUBPDr213mY VR256:$src1, VR256:$src2, addr:$src3)>;
  def : Pat<(int_x86_fma_vfmaddsub_pd_256 VR256:$src2, VR256:$src1, VR256:$src3),
            (VFMADDSUBPDr213rY VR256:$src1, VR256:$src2, VR256:$src3)>;
  def : Pat<(int_x86_fma_vfmaddsub_pd_256 VR256:$src2, VR256:$src1,
             (memopv4f64 addr:$src3)),
            (VFMADDSUBPDr213mY VR256:$src1, VR256:$src2, addr:$src3)>;
  def : Pat<(int_x86_fma_vfmsubadd_pd_256 VR256:$src2, VR256:$src1, VR256:$src3),
            (VFMSUBADDPDr213rY VR256:$src1, VR256:$src2, VR256:$src3)>;
  def : Pat<(int_x86_fma_vfmsubadd_pd_256 VR256:$src2, VR256:$src1,
             (memopv4f64 addr:$src3)),
            (VFMSUBADDPDr213mY VR256:$src1, VR256:$src2, addr:$src3)>;

  def : Pat<(int_x86_fma_vfnmadd_ps VR128:$src2, VR128:$src1, VR128:$src3),
            (VFNMADDPSr213r VR128:$src1, VR128:$src2, VR128:$src3)>;
  def : Pat<(int_x86_fma_vfnmadd_ps VR128:$src2, VR128:$src1,
             (memopv4f32 addr:$src3)),
            (VFNMADDPSr213m VR128:$src1, VR128:$src2, addr:$src3)>;
  def : Pat<(int_x86_fma_vfnmsub_ps VR128:$src2, VR128:$src1, VR128:$src3),
            (VFNMSUBPSr213r VR128:$src1, VR128:$src2, VR128:$src3)>;
  def : Pat<(int_x86_fma_vfnmsub_ps VR128:$src2, VR128:$src1,
             (memopv4f32 addr:$src3)),
            (VFNMSUBPSr213m VR128:$src1, VR128:$src2, addr:$src3)>;

  def : Pat<(int_x86_fma_vfnmadd_ps_256 VR256:$src2, VR256:$src1, VR256:$src3),
            (VFNMADDPSr213rY VR256:$src1, VR256:$src2, VR256:$src3)>;
  def : Pat<(int_x86_fma_vfnmadd_ps_256 VR256:$src2, VR256:$src1,
             (memopv8f32 addr:$src3)),
            (VFNMADDPSr213mY VR256:$src1, VR256:$src2, addr:$src3)>;
  def : Pat<(int_x86_fma_vfnmsub_ps_256 VR256:$src2, VR256:$src1, VR256:$src3),
            (VFNMSUBPSr213rY VR256:$src1, VR256:$src2, VR256:$src3)>;
  def : Pat<(int_x86_fma_vfnmsub_ps_256 VR256:$src2, VR256:$src1,
             (memopv8f32 addr:$src3)),
            (VFNMSUBPSr213mY VR256:$src1, VR256:$src2, addr:$src3)>;

  def : Pat<(int_x86_fma_vfnmadd_pd VR128:$src2, VR128:$src1, VR128:$src3),
            (VFNMADDPDr213r VR128:$src1, VR128:$src2, VR128:$src3)>;
  def : Pat<(int_x86_fma_vfnmadd_pd VR128:$src2, VR128:$src1,
             (memopv2f64 addr:$src3)),
            (VFNMADDPDr213m VR128:$src1, VR128:$src2, addr:$src3)>;
  def : Pat<(int_x86_fma_vfnmsub_pd VR128:$src2, VR128:$src1, VR128:$src3),
            (VFNMSUBPDr213r VR128:$src1, VR128:$src2, VR128:$src3)>;
  def : Pat<(int_x86_fma_vfnmsub_pd VR128:$src2, VR128:$src1,
             (memopv2f64 addr:$src3)),
            (VFNMSUBPDr213m VR128:$src1, VR128:$src2, addr:$src3)>;

  def : Pat<(int_x86_fma_vfnmadd_pd_256 VR256:$src2, VR256:$src1, VR256:$src3),
            (VFNMADDPDr213rY VR256:$src1, VR256:$src2, VR256:$src3)>;
  def : Pat<(int_x86_fma_vfnmadd_pd_256 VR256:$src2, VR256:$src1,
             (memopv4f64 addr:$src3)),
            (VFNMADDPDr213mY VR256:$src1, VR256:$src2, addr:$src3)>;
  def : Pat<(int_x86_fma_vfnmsub_pd_256 VR256:$src2, VR256:$src1, VR256:$src3),
            (VFNMSUBPDr213rY VR256:$src1, VR256:$src2, VR256:$src3)>;
  def : Pat<(int_x86_fma_vfnmsub_pd_256 VR256:$src2, VR256:$src1,
             (memopv4f64 addr:$src3)),
            (VFNMSUBPDr213mY VR256:$src1, VR256:$src2, addr:$src3)>;

} // Predicates = [HasFMA]

let Constraints = "$src1 = $dst" in {
multiclass fma3s_rm<bits<8> opc, string OpcodeStr, X86MemOperand x86memop,
                    RegisterClass RC, ValueType OpVT, PatFrag mem_frag,
                    SDPatternOperator OpNode = null_frag, bit MayLoad = 1> {
  def r     : FMA3<opc, MRMSrcReg, (outs RC:$dst),
                   (ins RC:$src1, RC:$src2, RC:$src3),
                   !strconcat(OpcodeStr,
                              "\t{$src3, $src2, $dst|$dst, $src2, $src3}"),
                   [(set RC:$dst,
                     (OpVT (OpNode RC:$src2, RC:$src1, RC:$src3)))]>;
  let mayLoad = MayLoad in
  def m     : FMA3<opc, MRMSrcMem, (outs RC:$dst),
                   (ins RC:$src1, RC:$src2, x86memop:$src3),
                   !strconcat(OpcodeStr,
                              "\t{$src3, $src2, $dst|$dst, $src2, $src3}"),
                   [(set RC:$dst,
                     (OpVT (OpNode RC:$src2, RC:$src1,
                            (mem_frag addr:$src3))))]>;
}

multiclass fma3s_rm_int<bits<8> opc, string OpcodeStr, Operand memop,
                        ComplexPattern mem_cpat, Intrinsic IntId,
                        RegisterClass RC> {
  def r_Int : FMA3<opc, MRMSrcReg, (outs VR128:$dst),
                   (ins VR128:$src1, VR128:$src2, VR128:$src3),
                   !strconcat(OpcodeStr,
                              "\t{$src3, $src2, $dst|$dst, $src2, $src3}"),
                   [(set VR128:$dst, (IntId VR128:$src2, VR128:$src1,
                     VR128:$src3))]>;
  def m_Int : FMA3<opc, MRMSrcMem, (outs VR128:$dst),
                   (ins VR128:$src1, VR128:$src2, memop:$src3),
                   !strconcat(OpcodeStr,
                              "\t{$src3, $src2, $dst|$dst, $src2, $src3}"),
                   [(set VR128:$dst,
                     (IntId VR128:$src2, VR128:$src1, mem_cpat:$src3))]>;
}
} // Constraints = "$src1 = $dst"

multiclass fma3s_forms<bits<8> opc132, bits<8> opc213, bits<8> opc231,
                       string OpStr, string PackTy, Intrinsic Int,
                       SDNode OpNode, RegisterClass RC, ValueType OpVT,
                       X86MemOperand x86memop, Operand memop, PatFrag mem_frag,
                       ComplexPattern mem_cpat> {
let neverHasSideEffects = 1 in {
  defm r132 : fma3s_rm<opc132, !strconcat(OpStr, !strconcat("132", PackTy)),
                       x86memop, RC, OpVT, mem_frag>;
  defm r231 : fma3s_rm<opc231, !strconcat(OpStr, !strconcat("231", PackTy)),
                       x86memop, RC, OpVT, mem_frag>;
}

defm r213 : fma3s_rm<opc213, !strconcat(OpStr, !strconcat("213", PackTy)),
                     x86memop, RC, OpVT, mem_frag, OpNode, 0>,
            fma3s_rm_int<opc213, !strconcat(OpStr, !strconcat("213", PackTy)),
                         memop, mem_cpat, Int, RC>;
}

multiclass fma3s<bits<8> opc132, bits<8> opc213, bits<8> opc231,
                 string OpStr, Intrinsic IntF32, Intrinsic IntF64,
                 SDNode OpNode> {
  defm SS : fma3s_forms<opc132, opc213, opc231, OpStr, "ss", IntF32, OpNode,
                        FR32, f32, f32mem, ssmem, loadf32, sse_load_f32>;
  defm SD : fma3s_forms<opc132, opc213, opc231, OpStr, "sd", IntF64, OpNode,
                        FR64, f64, f64mem, sdmem, loadf64, sse_load_f64>, VEX_W;
}

defm VFMADD : fma3s<0x99, 0xA9, 0xB9, "vfmadd", int_x86_fma_vfmadd_ss,
                    int_x86_fma_vfmadd_sd, X86Fmadd>, VEX_LIG;
defm VFMSUB : fma3s<0x9B, 0xAB, 0xBB, "vfmsub", int_x86_fma_vfmsub_ss,
                    int_x86_fma_vfmsub_sd, X86Fmsub>, VEX_LIG;

defm VFNMADD : fma3s<0x9D, 0xAD, 0xBD, "vfnmadd", int_x86_fma_vfnmadd_ss,
                     int_x86_fma_vfnmadd_sd, X86Fnmadd>, VEX_LIG;
defm VFNMSUB : fma3s<0x9F, 0xAF, 0xBF, "vfnmsub", int_x86_fma_vfnmsub_ss,
                     int_x86_fma_vfnmsub_sd, X86Fnmsub>, VEX_LIG;


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

