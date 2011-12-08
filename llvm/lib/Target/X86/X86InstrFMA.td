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
           "\t{$src3, $src2, $src1, $dst|$dst, $src1, $src2, $src3}"),
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

multiclass fma4p<bits<8> opc, string OpcodeStr> {
  def rr : FMA4<opc, MRMSrcReg, (outs VR128:$dst),
           (ins VR128:$src1, VR128:$src2, VR128:$src3),
           !strconcat(OpcodeStr,
           "\t{$src3, $src2, $src1, $dst|$dst, $src1, $src2, $src3}"),
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
  def rrY : FMA4<opc, MRMSrcReg, (outs VR256:$dst),
           (ins VR256:$src1, VR256:$src2, VR256:$src3),
           !strconcat(OpcodeStr,
           "\t{$src3, $src2, $src1, $dst|$dst, $src1, $src2, $src3}"),
           []>, XOP_W;
  def rmY : FMA4<opc, MRMSrcMem, (outs VR256:$dst),
           (ins VR256:$src1, VR256:$src2, f256mem:$src3),
           !strconcat(OpcodeStr,
           "\t{$src3, $src2, $src1, $dst|$dst, $src1, $src2, $src3}"),
           []>, XOP_W;
  def mrY : FMA4<opc, MRMSrcMem, (outs VR256:$dst),
           (ins VR256:$src1, f256mem:$src2, VR256:$src3),
           !strconcat(OpcodeStr,
           "\t{$src3, $src2, $src1, $dst|$dst, $src1, $src2, $src3}"),
           []>;
}

let isAsmParserOnly = 1 in {
  defm VFMADDSS4    : fma4s<0x6A, "vfmaddss">;
  defm VFMADDSD4    : fma4s<0x6B, "vfmaddsd">;
  defm VFMADDPS4    : fma4p<0x68, "vfmaddps">;
  defm VFMADDPD4    : fma4p<0x69, "vfmaddpd">;
  defm VFMSUBSS4    : fma4s<0x6E, "vfmsubss">;
  defm VFMSUBSD4    : fma4s<0x6F, "vfmsubsd">;
  defm VFMSUBPS4    : fma4p<0x6C, "vfmsubps">;
  defm VFMSUBPD4    : fma4p<0x6D, "vfmsubpd">;
  defm VFNMADDSS4   : fma4s<0x7A, "vfnmaddss">;
  defm VFNMADDSD4   : fma4s<0x7B, "vfnmaddsd">;
  defm VFNMADDPS4   : fma4p<0x78, "vfnmaddps">;
  defm VFNMADDPD4   : fma4p<0x79, "vfnmaddpd">;
  defm VFNMSUBSS4   : fma4s<0x7E, "vfnmsubss">;
  defm VFNMSUBSD4   : fma4s<0x7F, "vfnmsubsd">;
  defm VFNMSUBPS4   : fma4p<0x7C, "vfnmsubps">;
  defm VFNMSUBPD4   : fma4p<0x7D, "vfnmsubpd">;
  defm VFMADDSUBPS4 : fma4p<0x5C, "vfmaddsubps">;
  defm VFMADDSUBPD4 : fma4p<0x5D, "vfmaddsubpd">;
  defm VFMSUBADDPS4 : fma4p<0x5E, "vfmsubaddps">;
  defm VFMSUBADDPD4 : fma4p<0x5F, "vfmsubaddpd">;
}

// FMA4 Intrinsics patterns

// VFMADD
def : Pat<(int_x86_fma4_vfmadd_ss VR128:$src1, VR128:$src2, VR128:$src3),
          (VFMADDSS4rr VR128:$src1, VR128:$src2, VR128:$src3)>;
def : Pat<(int_x86_fma4_vfmadd_ss VR128:$src1, VR128:$src2,
                                  (alignedloadv4f32 addr:$src3)),
          (VFMADDSS4rm VR128:$src1, VR128:$src2, addr:$src3)>;
def : Pat<(int_x86_fma4_vfmadd_ss VR128:$src1, (alignedloadv4f32 addr:$src2),
                                  VR128:$src3),
          (VFMADDSS4mr VR128:$src1, addr:$src2, VR128:$src3)>;

def : Pat<(int_x86_fma4_vfmadd_sd VR128:$src1, VR128:$src2, VR128:$src3),
          (VFMADDSD4rr VR128:$src1, VR128:$src2, VR128:$src3)>;
def : Pat<(int_x86_fma4_vfmadd_sd VR128:$src1, VR128:$src2,
                                  (alignedloadv2f64 addr:$src3)),
          (VFMADDSD4rm VR128:$src1, VR128:$src2, addr:$src3)>;
def : Pat<(int_x86_fma4_vfmadd_sd VR128:$src1, (alignedloadv2f64 addr:$src2),
                                  VR128:$src3),
          (VFMADDSD4mr VR128:$src1, addr:$src2, VR128:$src3)>;

def : Pat<(int_x86_fma4_vfmadd_ps VR128:$src1, VR128:$src2, VR128:$src3),
          (VFMADDPS4rr VR128:$src1, VR128:$src2, VR128:$src3)>;
def : Pat<(int_x86_fma4_vfmadd_ps VR128:$src1, VR128:$src2,
                                  (alignedloadv4f32 addr:$src3)),
          (VFMADDPS4rm VR128:$src1, VR128:$src2, addr:$src3)>;
def : Pat<(int_x86_fma4_vfmadd_ps VR128:$src1, (alignedloadv4f32 addr:$src2),
                                  VR128:$src3),
          (VFMADDPS4mr VR128:$src1, addr:$src2, VR128:$src3)>;

def : Pat<(int_x86_fma4_vfmadd_pd VR128:$src1, VR128:$src2, VR128:$src3),
          (VFMADDPD4rr VR128:$src1, VR128:$src2, VR128:$src3)>;
def : Pat<(int_x86_fma4_vfmadd_pd VR128:$src1, VR128:$src2,
                                  (alignedloadv2f64 addr:$src3)),
          (VFMADDPD4rm VR128:$src1, VR128:$src2, addr:$src3)>;
def : Pat<(int_x86_fma4_vfmadd_pd VR128:$src1, (alignedloadv2f64 addr:$src2),
                                  VR128:$src3),
          (VFMADDPD4mr VR128:$src1, addr:$src2, VR128:$src3)>;

def : Pat<(int_x86_fma4_vfmadd_ps_256 VR256:$src1, VR256:$src2, VR256:$src3),
          (VFMADDPS4rrY VR256:$src1, VR256:$src2, VR256:$src3)>;
def : Pat<(int_x86_fma4_vfmadd_ps_256 VR256:$src1, VR256:$src2,
                                  (alignedloadv8f32 addr:$src3)),
          (VFMADDPS4rmY VR256:$src1, VR256:$src2, addr:$src3)>;
def : Pat<(int_x86_fma4_vfmadd_ps_256 VR256:$src1,
                                      (alignedloadv8f32 addr:$src2),
                                      VR256:$src3),
          (VFMADDPS4mrY VR256:$src1, addr:$src2, VR256:$src3)>;

def : Pat<(int_x86_fma4_vfmadd_pd_256 VR256:$src1, VR256:$src2, VR256:$src3),
          (VFMADDPD4rrY VR256:$src1, VR256:$src2, VR256:$src3)>;
def : Pat<(int_x86_fma4_vfmadd_pd_256 VR256:$src1, VR256:$src2,
                                  (alignedloadv4f64 addr:$src3)),
          (VFMADDPD4rmY VR256:$src1, VR256:$src2, addr:$src3)>;
def : Pat<(int_x86_fma4_vfmadd_pd_256 VR256:$src1,
                                      (alignedloadv4f64 addr:$src2),
                                      VR256:$src3),
          (VFMADDPD4mrY VR256:$src1, addr:$src2, VR256:$src3)>;

// VFMSUB
def : Pat<(int_x86_fma4_vfmsub_ss VR128:$src1, VR128:$src2, VR128:$src3),
          (VFMSUBSS4rr VR128:$src1, VR128:$src2, VR128:$src3)>;
def : Pat<(int_x86_fma4_vfmsub_ss VR128:$src1, VR128:$src2,
                                  (alignedloadv4f32 addr:$src3)),
          (VFMSUBSS4rm VR128:$src1, VR128:$src2, addr:$src3)>;
def : Pat<(int_x86_fma4_vfmsub_ss VR128:$src1, (alignedloadv4f32 addr:$src2),
                                  VR128:$src3),
          (VFMSUBSS4mr VR128:$src1, addr:$src2, VR128:$src3)>;

def : Pat<(int_x86_fma4_vfmsub_sd VR128:$src1, VR128:$src2, VR128:$src3),
          (VFMSUBSD4rr VR128:$src1, VR128:$src2, VR128:$src3)>;
def : Pat<(int_x86_fma4_vfmsub_sd VR128:$src1, VR128:$src2,
                                  (alignedloadv2f64 addr:$src3)),
          (VFMSUBSD4rm VR128:$src1, VR128:$src2, addr:$src3)>;
def : Pat<(int_x86_fma4_vfmsub_sd VR128:$src1, (alignedloadv2f64 addr:$src2),
                                  VR128:$src3),
          (VFMSUBSD4mr VR128:$src1, addr:$src2, VR128:$src3)>;

def : Pat<(int_x86_fma4_vfmsub_ps VR128:$src1, VR128:$src2, VR128:$src3),
          (VFMSUBPS4rr VR128:$src1, VR128:$src2, VR128:$src3)>;
def : Pat<(int_x86_fma4_vfmsub_ps VR128:$src1, VR128:$src2,
                                  (alignedloadv4f32 addr:$src3)),
          (VFMSUBPS4rm VR128:$src1, VR128:$src2, addr:$src3)>;
def : Pat<(int_x86_fma4_vfmsub_ps VR128:$src1, (alignedloadv4f32 addr:$src2),
                                  VR128:$src3),
          (VFMSUBPS4mr VR128:$src1, addr:$src2, VR128:$src3)>;

def : Pat<(int_x86_fma4_vfmsub_pd VR128:$src1, VR128:$src2, VR128:$src3),
          (VFMSUBPD4rr VR128:$src1, VR128:$src2, VR128:$src3)>;
def : Pat<(int_x86_fma4_vfmsub_pd VR128:$src1, VR128:$src2,
                                  (alignedloadv2f64 addr:$src3)),
          (VFMSUBPD4rm VR128:$src1, VR128:$src2, addr:$src3)>;
def : Pat<(int_x86_fma4_vfmsub_pd VR128:$src1, (alignedloadv2f64 addr:$src2),
                                  VR128:$src3),
          (VFMSUBPD4mr VR128:$src1, addr:$src2, VR128:$src3)>;

def : Pat<(int_x86_fma4_vfmsub_ps_256 VR256:$src1, VR256:$src2, VR256:$src3),
          (VFMSUBPS4rrY VR256:$src1, VR256:$src2, VR256:$src3)>;
def : Pat<(int_x86_fma4_vfmsub_ps_256 VR256:$src1, VR256:$src2,
                                  (alignedloadv8f32 addr:$src3)),
          (VFMSUBPS4rmY VR256:$src1, VR256:$src2, addr:$src3)>;
def : Pat<(int_x86_fma4_vfmsub_ps_256 VR256:$src1,
                                      (alignedloadv8f32 addr:$src2),
                                      VR256:$src3),
          (VFMSUBPS4mrY VR256:$src1, addr:$src2, VR256:$src3)>;

def : Pat<(int_x86_fma4_vfmsub_pd_256 VR256:$src1, VR256:$src2, VR256:$src3),
          (VFMSUBPD4rrY VR256:$src1, VR256:$src2, VR256:$src3)>;
def : Pat<(int_x86_fma4_vfmsub_pd_256 VR256:$src1, VR256:$src2,
                                  (alignedloadv4f64 addr:$src3)),
          (VFMSUBPD4rmY VR256:$src1, VR256:$src2, addr:$src3)>;
def : Pat<(int_x86_fma4_vfmsub_pd_256 VR256:$src1,
                                      (alignedloadv4f64 addr:$src2),
                                      VR256:$src3),
          (VFMSUBPD4mrY VR256:$src1, addr:$src2, VR256:$src3)>;

// VFNMADD
def : Pat<(int_x86_fma4_vfnmadd_ss VR128:$src1, VR128:$src2, VR128:$src3),
          (VFNMADDSS4rr VR128:$src1, VR128:$src2, VR128:$src3)>;
def : Pat<(int_x86_fma4_vfnmadd_ss VR128:$src1, VR128:$src2,
                                  (alignedloadv4f32 addr:$src3)),
          (VFNMADDSS4rm VR128:$src1, VR128:$src2, addr:$src3)>;
def : Pat<(int_x86_fma4_vfnmadd_ss VR128:$src1, (alignedloadv4f32 addr:$src2),
                                  VR128:$src3),
          (VFNMADDSS4mr VR128:$src1, addr:$src2, VR128:$src3)>;

def : Pat<(int_x86_fma4_vfnmadd_sd VR128:$src1, VR128:$src2, VR128:$src3),
          (VFNMADDSD4rr VR128:$src1, VR128:$src2, VR128:$src3)>;
def : Pat<(int_x86_fma4_vfnmadd_sd VR128:$src1, VR128:$src2,
                                  (alignedloadv2f64 addr:$src3)),
          (VFNMADDSD4rm VR128:$src1, VR128:$src2, addr:$src3)>;
def : Pat<(int_x86_fma4_vfnmadd_sd VR128:$src1, (alignedloadv2f64 addr:$src2),
                                  VR128:$src3),
          (VFNMADDSD4mr VR128:$src1, addr:$src2, VR128:$src3)>;

def : Pat<(int_x86_fma4_vfnmadd_ps VR128:$src1, VR128:$src2, VR128:$src3),
          (VFNMADDPS4rr VR128:$src1, VR128:$src2, VR128:$src3)>;
def : Pat<(int_x86_fma4_vfnmadd_ps VR128:$src1, VR128:$src2,
                                  (alignedloadv4f32 addr:$src3)),
          (VFNMADDPS4rm VR128:$src1, VR128:$src2, addr:$src3)>;
def : Pat<(int_x86_fma4_vfnmadd_ps VR128:$src1, (alignedloadv4f32 addr:$src2),
                                  VR128:$src3),
          (VFNMADDPS4mr VR128:$src1, addr:$src2, VR128:$src3)>;

def : Pat<(int_x86_fma4_vfnmadd_pd VR128:$src1, VR128:$src2, VR128:$src3),
          (VFNMADDPD4rr VR128:$src1, VR128:$src2, VR128:$src3)>;
def : Pat<(int_x86_fma4_vfnmadd_pd VR128:$src1, VR128:$src2,
                                  (alignedloadv2f64 addr:$src3)),
          (VFNMADDPD4rm VR128:$src1, VR128:$src2, addr:$src3)>;
def : Pat<(int_x86_fma4_vfnmadd_pd VR128:$src1, (alignedloadv2f64 addr:$src2),
                                  VR128:$src3),
          (VFNMADDPD4mr VR128:$src1, addr:$src2, VR128:$src3)>;

def : Pat<(int_x86_fma4_vfnmadd_ps_256 VR256:$src1, VR256:$src2, VR256:$src3),
          (VFNMADDPS4rrY VR256:$src1, VR256:$src2, VR256:$src3)>;
def : Pat<(int_x86_fma4_vfnmadd_ps_256 VR256:$src1, VR256:$src2,
                                  (alignedloadv8f32 addr:$src3)),
          (VFNMADDPS4rmY VR256:$src1, VR256:$src2, addr:$src3)>;
def : Pat<(int_x86_fma4_vfnmadd_ps_256 VR256:$src1,
                                      (alignedloadv8f32 addr:$src2),
                                      VR256:$src3),
          (VFNMADDPS4mrY VR256:$src1, addr:$src2, VR256:$src3)>;

def : Pat<(int_x86_fma4_vfnmadd_pd_256 VR256:$src1, VR256:$src2, VR256:$src3),
          (VFNMADDPD4rrY VR256:$src1, VR256:$src2, VR256:$src3)>;
def : Pat<(int_x86_fma4_vfnmadd_pd_256 VR256:$src1, VR256:$src2,
                                  (alignedloadv4f64 addr:$src3)),
          (VFNMADDPD4rmY VR256:$src1, VR256:$src2, addr:$src3)>;
def : Pat<(int_x86_fma4_vfnmadd_pd_256 VR256:$src1,
                                      (alignedloadv4f64 addr:$src2),
                                      VR256:$src3),
          (VFNMADDPD4mrY VR256:$src1, addr:$src2, VR256:$src3)>;

// VFNMSUB
def : Pat<(int_x86_fma4_vfnmsub_ss VR128:$src1, VR128:$src2, VR128:$src3),
          (VFNMSUBSS4rr VR128:$src1, VR128:$src2, VR128:$src3)>;
def : Pat<(int_x86_fma4_vfnmsub_ss VR128:$src1, VR128:$src2,
                                  (alignedloadv4f32 addr:$src3)),
          (VFNMSUBSS4rm VR128:$src1, VR128:$src2, addr:$src3)>;
def : Pat<(int_x86_fma4_vfnmsub_ss VR128:$src1, (alignedloadv4f32 addr:$src2),
                                  VR128:$src3),
          (VFNMSUBSS4mr VR128:$src1, addr:$src2, VR128:$src3)>;

def : Pat<(int_x86_fma4_vfnmsub_sd VR128:$src1, VR128:$src2, VR128:$src3),
          (VFNMSUBSD4rr VR128:$src1, VR128:$src2, VR128:$src3)>;
def : Pat<(int_x86_fma4_vfnmsub_sd VR128:$src1, VR128:$src2,
                                  (alignedloadv2f64 addr:$src3)),
          (VFNMSUBSD4rm VR128:$src1, VR128:$src2, addr:$src3)>;
def : Pat<(int_x86_fma4_vfnmsub_sd VR128:$src1, (alignedloadv2f64 addr:$src2),
                                  VR128:$src3),
          (VFNMSUBSD4mr VR128:$src1, addr:$src2, VR128:$src3)>;

def : Pat<(int_x86_fma4_vfnmsub_ps VR128:$src1, VR128:$src2, VR128:$src3),
          (VFNMSUBPS4rr VR128:$src1, VR128:$src2, VR128:$src3)>;
def : Pat<(int_x86_fma4_vfnmsub_ps VR128:$src1, VR128:$src2,
                                  (alignedloadv4f32 addr:$src3)),
          (VFNMSUBPS4rm VR128:$src1, VR128:$src2, addr:$src3)>;
def : Pat<(int_x86_fma4_vfnmsub_ps VR128:$src1, (alignedloadv4f32 addr:$src2),
                                  VR128:$src3),
          (VFNMSUBPS4mr VR128:$src1, addr:$src2, VR128:$src3)>;

def : Pat<(int_x86_fma4_vfnmsub_pd VR128:$src1, VR128:$src2, VR128:$src3),
          (VFNMSUBPD4rr VR128:$src1, VR128:$src2, VR128:$src3)>;
def : Pat<(int_x86_fma4_vfnmsub_pd VR128:$src1, VR128:$src2,
                                  (alignedloadv2f64 addr:$src3)),
          (VFNMSUBPD4rm VR128:$src1, VR128:$src2, addr:$src3)>;
def : Pat<(int_x86_fma4_vfnmsub_pd VR128:$src1, (alignedloadv2f64 addr:$src2),
                                  VR128:$src3),
          (VFNMSUBPD4mr VR128:$src1, addr:$src2, VR128:$src3)>;

def : Pat<(int_x86_fma4_vfnmsub_ps_256 VR256:$src1, VR256:$src2, VR256:$src3),
          (VFNMSUBPS4rrY VR256:$src1, VR256:$src2, VR256:$src3)>;
def : Pat<(int_x86_fma4_vfnmsub_ps_256 VR256:$src1, VR256:$src2,
                                  (alignedloadv8f32 addr:$src3)),
          (VFNMSUBPS4rmY VR256:$src1, VR256:$src2, addr:$src3)>;
def : Pat<(int_x86_fma4_vfnmsub_ps_256 VR256:$src1,
                                      (alignedloadv8f32 addr:$src2),
                                      VR256:$src3),
          (VFNMSUBPS4mrY VR256:$src1, addr:$src2, VR256:$src3)>;

def : Pat<(int_x86_fma4_vfnmsub_pd_256 VR256:$src1, VR256:$src2, VR256:$src3),
          (VFNMSUBPD4rrY VR256:$src1, VR256:$src2, VR256:$src3)>;
def : Pat<(int_x86_fma4_vfnmsub_pd_256 VR256:$src1, VR256:$src2,
                                  (alignedloadv4f64 addr:$src3)),
          (VFNMSUBPD4rmY VR256:$src1, VR256:$src2, addr:$src3)>;
def : Pat<(int_x86_fma4_vfnmsub_pd_256 VR256:$src1,
                                      (alignedloadv4f64 addr:$src2),
                                      VR256:$src3),
          (VFNMSUBPD4mrY VR256:$src1, addr:$src2, VR256:$src3)>;

// VFMADDSUB
def : Pat<(int_x86_fma4_vfmaddsub_ps VR128:$src1, VR128:$src2, VR128:$src3),
          (VFMADDSUBPS4rr VR128:$src1, VR128:$src2, VR128:$src3)>;
def : Pat<(int_x86_fma4_vfmaddsub_ps VR128:$src1, VR128:$src2,
                                  (alignedloadv4f32 addr:$src3)),
          (VFMADDSUBPS4rm VR128:$src1, VR128:$src2, addr:$src3)>;
def : Pat<(int_x86_fma4_vfmaddsub_ps VR128:$src1, (alignedloadv4f32 addr:$src2),
                                  VR128:$src3),
          (VFMADDSUBPS4mr VR128:$src1, addr:$src2, VR128:$src3)>;

def : Pat<(int_x86_fma4_vfmaddsub_pd VR128:$src1, VR128:$src2, VR128:$src3),
          (VFMADDSUBPD4rr VR128:$src1, VR128:$src2, VR128:$src3)>;
def : Pat<(int_x86_fma4_vfmaddsub_pd VR128:$src1, VR128:$src2,
                                  (alignedloadv2f64 addr:$src3)),
          (VFMADDSUBPD4rm VR128:$src1, VR128:$src2, addr:$src3)>;
def : Pat<(int_x86_fma4_vfmaddsub_pd VR128:$src1, (alignedloadv2f64 addr:$src2),
                                  VR128:$src3),
          (VFMADDSUBPD4mr VR128:$src1, addr:$src2, VR128:$src3)>;

def : Pat<(int_x86_fma4_vfmaddsub_ps_256 VR256:$src1, VR256:$src2, VR256:$src3),
          (VFMADDSUBPS4rrY VR256:$src1, VR256:$src2, VR256:$src3)>;
def : Pat<(int_x86_fma4_vfmaddsub_ps_256 VR256:$src1, VR256:$src2,
                                  (alignedloadv8f32 addr:$src3)),
          (VFMADDSUBPS4rmY VR256:$src1, VR256:$src2, addr:$src3)>;
def : Pat<(int_x86_fma4_vfmaddsub_ps_256 VR256:$src1,
                                      (alignedloadv8f32 addr:$src2),
                                      VR256:$src3),
          (VFMADDSUBPS4mrY VR256:$src1, addr:$src2, VR256:$src3)>;

def : Pat<(int_x86_fma4_vfmaddsub_pd_256 VR256:$src1, VR256:$src2, VR256:$src3),
          (VFMADDSUBPD4rrY VR256:$src1, VR256:$src2, VR256:$src3)>;
def : Pat<(int_x86_fma4_vfmaddsub_pd_256 VR256:$src1, VR256:$src2,
                                  (alignedloadv4f64 addr:$src3)),
          (VFMADDSUBPD4rmY VR256:$src1, VR256:$src2, addr:$src3)>;
def : Pat<(int_x86_fma4_vfmaddsub_pd_256 VR256:$src1,
                                      (alignedloadv4f64 addr:$src2),
                                      VR256:$src3),
          (VFMADDSUBPD4mrY VR256:$src1, addr:$src2, VR256:$src3)>;

// VFMSUBADD
def : Pat<(int_x86_fma4_vfmsubadd_ps VR128:$src1, VR128:$src2, VR128:$src3),
          (VFMSUBADDPS4rr VR128:$src1, VR128:$src2, VR128:$src3)>;
def : Pat<(int_x86_fma4_vfmsubadd_ps VR128:$src1, VR128:$src2,
                                  (alignedloadv4f32 addr:$src3)),
          (VFMSUBADDPS4rm VR128:$src1, VR128:$src2, addr:$src3)>;
def : Pat<(int_x86_fma4_vfmsubadd_ps VR128:$src1, (alignedloadv4f32 addr:$src2),
                                  VR128:$src3),
          (VFMSUBADDPS4mr VR128:$src1, addr:$src2, VR128:$src3)>;

def : Pat<(int_x86_fma4_vfmsubadd_pd VR128:$src1, VR128:$src2, VR128:$src3),
          (VFMSUBADDPD4rr VR128:$src1, VR128:$src2, VR128:$src3)>;
def : Pat<(int_x86_fma4_vfmsubadd_pd VR128:$src1, VR128:$src2,
                                  (alignedloadv2f64 addr:$src3)),
          (VFMSUBADDPD4rm VR128:$src1, VR128:$src2, addr:$src3)>;
def : Pat<(int_x86_fma4_vfmsubadd_pd VR128:$src1, (alignedloadv2f64 addr:$src2),
                                  VR128:$src3),
          (VFMSUBADDPD4mr VR128:$src1, addr:$src2, VR128:$src3)>;

def : Pat<(int_x86_fma4_vfmsubadd_ps_256 VR256:$src1, VR256:$src2, VR256:$src3),
          (VFMSUBADDPS4rrY VR256:$src1, VR256:$src2, VR256:$src3)>;
def : Pat<(int_x86_fma4_vfmsubadd_ps_256 VR256:$src1, VR256:$src2,
                                  (alignedloadv8f32 addr:$src3)),
          (VFMSUBADDPS4rmY VR256:$src1, VR256:$src2, addr:$src3)>;
def : Pat<(int_x86_fma4_vfmsubadd_ps_256 VR256:$src1,
                                      (alignedloadv8f32 addr:$src2),
                                      VR256:$src3),
          (VFMSUBADDPS4mrY VR256:$src1, addr:$src2, VR256:$src3)>;

def : Pat<(int_x86_fma4_vfmsubadd_pd_256 VR256:$src1, VR256:$src2, VR256:$src3),
          (VFMSUBADDPD4rrY VR256:$src1, VR256:$src2, VR256:$src3)>;
def : Pat<(int_x86_fma4_vfmsubadd_pd_256 VR256:$src1, VR256:$src2,
                                  (alignedloadv4f64 addr:$src3)),
          (VFMSUBADDPD4rmY VR256:$src1, VR256:$src2, addr:$src3)>;
def : Pat<(int_x86_fma4_vfmsubadd_pd_256 VR256:$src1,
                                      (alignedloadv4f64 addr:$src2),
                                      VR256:$src3),
          (VFMSUBADDPD4mrY VR256:$src1, addr:$src2, VR256:$src3)>;
