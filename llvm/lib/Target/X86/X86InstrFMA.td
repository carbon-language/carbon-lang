//====- X86InstrFMA.td - Describe the X86 Instruction Set --*- tablegen -*-===//
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

multiclass fma_rm<bits<8> opc, string OpcodeStr> {
  def r : FMA3<opc, MRMSrcReg, (outs VR128:$dst),
           (ins VR128:$src1, VR128:$src2),
           !strconcat(OpcodeStr, "\t{$src2, $src1, $dst|$dst, $src1, $src2}"),
           []>;
  def m : FMA3<opc, MRMSrcMem, (outs VR128:$dst),
           (ins VR128:$src1, f128mem:$src2),
           !strconcat(OpcodeStr, "\t{$src2, $src1, $dst|$dst, $src1, $src2}"),
           []>;
  def rY : FMA3<opc, MRMSrcReg, (outs VR256:$dst),
           (ins VR256:$src1, VR256:$src2),
           !strconcat(OpcodeStr, "\t{$src2, $src1, $dst|$dst, $src1, $src2}"),
           []>;
  def mY : FMA3<opc, MRMSrcMem, (outs VR256:$dst),
           (ins VR256:$src1, f256mem:$src2),
           !strconcat(OpcodeStr, "\t{$src2, $src1, $dst|$dst, $src1, $src2}"),
           []>;
}

multiclass fma_forms<bits<8> opc132, bits<8> opc213, bits<8> opc231,
                     string OpcodeStr, string PackTy> {
  defm r132 : fma_rm<opc132, !strconcat(OpcodeStr, !strconcat("132", PackTy))>;
  defm r213 : fma_rm<opc213, !strconcat(OpcodeStr, !strconcat("213", PackTy))>;
  defm r231 : fma_rm<opc231, !strconcat(OpcodeStr, !strconcat("231", PackTy))>;
}

let isAsmParserOnly = 1 in {
  // Fused Multiply-Add
  defm VFMADDPS    : fma_forms<0x98, 0xA8, 0xB8, "vfmadd", "ps">;
  defm VFMADDPD    : fma_forms<0x98, 0xA8, 0xB8, "vfmadd", "pd">, VEX_W;
  defm VFMADDSUBPS : fma_forms<0x96, 0xA6, 0xB6, "vfmaddsub", "ps">;
  defm VFMADDSUBPD : fma_forms<0x96, 0xA6, 0xB6, "vfmaddsub", "pd">, VEX_W;
  defm VFMSUBADDPS : fma_forms<0x97, 0xA7, 0xB7, "vfmsubadd", "ps">;
  defm VFMSUBADDPD : fma_forms<0x97, 0xA7, 0xB7, "vfmsubadd", "pd">, VEX_W;
  defm VFMSUBPS    : fma_forms<0x9A, 0xAA, 0xBA, "vfmsub", "ps">;
  defm VFMSUBPD    : fma_forms<0x9A, 0xAA, 0xBA, "vfmsub", "pd">, VEX_W;

  // Fused Negative Multiply-Add
  defm VFNMADDPS : fma_forms<0x9C, 0xAC, 0xBC, "vfnmadd", "ps">;
  defm VFNMADDPD : fma_forms<0x9C, 0xAC, 0xBC, "vfnmadd", "pd">, VEX_W;
  defm VFNMSUBPS : fma_forms<0x9E, 0xAE, 0xBE, "vfnmsub", "ps">;
  defm VFNMSUBPD : fma_forms<0x9E, 0xAE, 0xBE, "vfnmsub", "pd">, VEX_W;
}

//===----------------------------------------------------------------------===//
// FMA4 - AMD 4 operand Fused Multiply-Add instructions
//===----------------------------------------------------------------------===//


multiclass fma4s<bits<8> opc, string OpcodeStr> {
  def rr : FMA4<opc, MRMSrcReg, (outs VR128:$dst),
           (ins VR128:$src1, VR128:$src2, VR128:$src3),
           !strconcat(OpcodeStr,
           "\t{$src2, $src3, $src1, $dst|$dst, $src1, $src3, $src2}"),
           []>, XOP_W;
  def rm : FMA4<opc, MRMSrcMem, (outs VR128:$dst),
           (ins VR128:$src1, VR128:$src2, f128mem:$src3),
           !strconcat(OpcodeStr,
           "\t{$src3, $src2, $src1, $dst|$dst, $src1, $src2, $src3}"),
           []>, XOP_W;
  def mr : FMA4<opc, MRMSrcMem, (outs VR128:$dst),
           (ins VR128:$src1, f128mem:$src2, VR128:$src3),
           !strconcat(OpcodeStr,
           "\t{$src3, $src2, $src1, $dst|$dst, $src1, $src2, $src3}"),
           []>;

}

let isAsmParserOnly = 1 in {
  defm VFMADDSD4    : fma4s<0x6B, "vfmaddsd">;
}

// FMA4 Intrinsics patterns

def : Pat<(int_x86_fma4_vfmadd_sd VR128:$src1, VR128:$src2, VR128:$src3),
          (VFMADDSD4rr VR128:$src1, VR128:$src2, VR128:$src3)>;
def : Pat<(int_x86_fma4_vfmadd_sd VR128:$src1, VR128:$src2,
                                  (alignedloadv2f64 addr:$src3)),
          (VFMADDSD4rm VR128:$src1, VR128:$src2, addr:$src3)>;
def : Pat<(int_x86_fma4_vfmadd_sd VR128:$src1, (alignedloadv2f64 addr:$src2),
                                  VR128:$src3),
          (VFMADDSD4mr VR128:$src1, addr:$src2, VR128:$src3)>;
