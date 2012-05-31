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
}

// Intrinsic for 132 pattern
multiclass fma3p_rm_int<bits<8> opc, string OpcodeStr, 
                        PatFrag MemFrag128, PatFrag MemFrag256,
                        Intrinsic Int128, Intrinsic Int256> {
  def r_Int : FMA3<opc, MRMSrcReg, (outs VR128:$dst),
           (ins VR128:$src1, VR128:$src2, VR128:$src3),
           !strconcat(OpcodeStr, "\t{$src3, $src2, $dst|$dst, $src2, $src3}"),
           [(set VR128:$dst, (Int128 VR128:$src1, VR128:$src3, VR128:$src2))]>;
  //let mayLoad = 1 in
  def m_Int : FMA3<opc, MRMSrcMem, (outs VR128:$dst),
           (ins VR128:$src1, VR128:$src2, f128mem:$src3),
           !strconcat(OpcodeStr, "\t{$src3, $src2, $dst|$dst, $src2, $src3}"),
           [(set VR128:$dst, (Int128 VR128:$src1, (MemFrag128 addr:$src3), VR128:$src2))]>;
  def rY_Int : FMA3<opc, MRMSrcReg, (outs VR256:$dst),
           (ins VR256:$src1, VR256:$src2, VR256:$src3),
           !strconcat(OpcodeStr, "\t{$src3, $src2, $dst|$dst, $src2, $src3}"),
           [(set VR256:$dst, (Int256 VR256:$src1, VR256:$src3, VR256:$src2))]>;
  //let mayLoad = 1 in
  def mY_Int : FMA3<opc, MRMSrcMem, (outs VR256:$dst),
           (ins VR256:$src1, VR256:$src2, f256mem:$src3),
           !strconcat(OpcodeStr, "\t{$src3, $src2, $dst|$dst, $src2, $src3}"),
           [(set VR256:$dst, (Int256 VR256:$src1, (MemFrag256 addr:$src3), VR256:$src2))]>;
}
}

multiclass fma3p_forms<bits<8> opc132, bits<8> opc213, bits<8> opc231,
                       string OpcodeStr, string PackTy, 
                       PatFrag MemFrag128, PatFrag MemFrag256,
                       Intrinsic Int128, Intrinsic Int256> {
  defm r132 : fma3p_rm_int <opc132, !strconcat(OpcodeStr, !strconcat("132", PackTy)),
                              MemFrag128, MemFrag256, Int128, Int256>;
  defm r132 : fma3p_rm     <opc132, !strconcat(OpcodeStr, !strconcat("132", PackTy))>;
  defm r213 : fma3p_rm     <opc213, !strconcat(OpcodeStr, !strconcat("213", PackTy))>;
  defm r231 : fma3p_rm     <opc231, !strconcat(OpcodeStr, !strconcat("231", PackTy))>;
}

// Fused Multiply-Add
let ExeDomain = SSEPackedSingle in {
  defm VFMADDPS    : fma3p_forms<0x98, 0xA8, 0xB8, "vfmadd", "ps", memopv4f32, memopv8f32,
    int_x86_fma4_vfmadd_ps, int_x86_fma4_vfmadd_ps_256>;
  defm VFMSUBPS    : fma3p_forms<0x9A, 0xAA, 0xBA, "vfmsub", "ps",  memopv4f32, memopv8f32,
    int_x86_fma4_vfmsub_ps, int_x86_fma4_vfmsub_ps_256>;
  defm VFMADDSUBPS : fma3p_forms<0x96, 0xA6, 0xB6, "vfmaddsub", "ps", memopv4f32, memopv8f32,
    int_x86_fma4_vfmaddsub_ps, int_x86_fma4_vfmaddsub_ps_256>;
  defm VFMSUBADDPS : fma3p_forms<0x97, 0xA7, 0xB7, "vfmsubadd", "ps", memopv4f32, memopv8f32,
    int_x86_fma4_vfmsubadd_ps, int_x86_fma4_vfmaddsub_ps_256>;
}

let ExeDomain = SSEPackedDouble in {
  defm VFMADDPD    : fma3p_forms<0x98, 0xA8, 0xB8, "vfmadd", "pd", memopv2f64, memopv4f64,
    int_x86_fma4_vfmadd_pd, int_x86_fma4_vfmadd_pd_256>, VEX_W;
  defm VFMSUBPD    : fma3p_forms<0x9A, 0xAA, 0xBA, "vfmsub", "pd", memopv2f64, memopv4f64,
    int_x86_fma4_vfmsub_pd, int_x86_fma4_vfmsub_pd_256>, VEX_W;
  defm VFMADDSUBPD : fma3p_forms<0x96, 0xA6, 0xB6, "vfmaddsub", "pd", memopv2f64, memopv4f64, 
    int_x86_fma4_vfmaddsub_pd, int_x86_fma4_vfmaddsub_pd_256>, VEX_W;
  defm VFMSUBADDPD : fma3p_forms<0x97, 0xA7, 0xB7, "vfmsubadd", "pd", memopv2f64, memopv4f64,
    int_x86_fma4_vfmsubadd_pd, int_x86_fma4_vfmsubadd_pd_256>, VEX_W;
}

// Fused Negative Multiply-Add
let ExeDomain = SSEPackedSingle in {
  defm VFNMADDPS : fma3p_forms<0x9C, 0xAC, 0xBC, "vfnmadd", "ps",  memopv4f32, memopv8f32,
    int_x86_fma4_vfnmadd_ps, int_x86_fma4_vfnmadd_ps_256>;
  defm VFNMSUBPS : fma3p_forms<0x9E, 0xAE, 0xBE, "vfnmsub", "ps",  memopv4f32, memopv8f32,
    int_x86_fma4_vfnmsub_ps, int_x86_fma4_vfnmsub_ps_256>;
}
let ExeDomain = SSEPackedDouble in {
  defm VFNMADDPD : fma3p_forms<0x9C, 0xAC, 0xBC, "vfnmadd", "pd", memopv2f64, memopv4f64,
    int_x86_fma4_vfnmadd_pd, int_x86_fma4_vfnmadd_pd_256>, VEX_W;
  defm VFNMSUBPD : fma3p_forms<0x9E, 0xAE, 0xBE, "vfnmsub", "pd", memopv2f64, memopv4f64,
    int_x86_fma4_vfnmsub_pd, int_x86_fma4_vfnmsub_pd_256>, VEX_W;
}

let Predicates = [HasFMA3], AddedComplexity = 20 in {
//------------
// FP double precision ADD - 256
//------------

// FMA231: src1 = src2*src3 + src1
def : Pat<(v4f64 (fadd (fmul VR256:$src2, (memopv4f64 addr:$src3)), VR256:$src1)),
        (VFMADDPDr231mY VR256:$src1, VR256:$src2, addr:$src3)>;
                        
// FMA231: src1 = src2*src3 + src1   
def : Pat<(v4f64 (fadd (fmul VR256:$src2, VR256:$src3), VR256:$src1)),
        (VFMADDPDr231rY VR256:$src1, VR256:$src2, VR256:$src3)>;
                        

//------------
// FP double precision ADD - 128
//------------


// FMA231: src1 = src2*src3 + src1
def : Pat<(v2f64 (fadd (fmul VR128:$src2, (memopv2f64 addr:$src3)), VR128:$src1)),
            (VFMADDPDr231m VR128:$src1, VR128:$src2, addr:$src3)>;
                        
// FMA231: src1 = src2*src3 + src1
def : Pat<(v2f64 (fadd (fmul VR128:$src2, VR128:$src3), VR128:$src1)),
            (VFMADDPDr231r VR128:$src1, VR128:$src2, VR128:$src3)>;

//------------
// FP double precision SUB - 256
//------------
// FMA231: src1 = src2*src3 - src1
def : Pat<(v4f64 (fsub (fmul VR256:$src2, (memopv4f64 addr:$src3)), VR256:$src1)),
           (VFMSUBPDr231mY VR256:$src1, VR256:$src2, addr:$src3)>;
                        
// FMA231: src1 = src2*src3 - src1
def : Pat<(v4f64 (fsub (fmul VR256:$src2, VR256:$src3), VR256:$src1)),
            (VFMSUBPDr231rY VR256:$src1, VR256:$src2, VR256:$src3)>;
                        

//------------
// FP double precision SUB - 128
//------------

// FMA231: src1 = src2*src3 - src1
def : Pat<(v2f64 (fsub (fmul VR128:$src2, (memopv2f64 addr:$src3)), VR128:$src1)),
            (VFMSUBPDr231m VR128:$src1, VR128:$src2, addr:$src3)>;
                        
// FMA231: src1 = src2*src3 - src1
def : Pat<(v2f64 (fsub (fmul VR128:$src2, VR128:$src3), VR128:$src1)),
            (VFMSUBPDr231r VR128:$src1, VR128:$src2, VR128:$src3)>;
                        
//------------
// FP double precision FNMADD - 256
//------------
// FMA231: src1 = - src2*src3 + src1
def : Pat<(v4f64 (fsub VR256:$src1, (fmul VR256:$src2, (memopv4f64 addr:$src3)))),
            (VFNMADDPDr231mY VR256:$src1, VR256:$src2, addr:$src3)>;
                        
// FMA231: src1 = - src2*src3 + src1
def : Pat<(v4f64 (fsub VR256:$src1, (fmul VR256:$src2, VR256:$src3))),
            (VFNMADDPDr231rY VR256:$src1, VR256:$src2, VR256:$src3)>;
                        
//------------
// FP double precision FNMADD - 128
//------------

// FMA231: src1 = - src2*src3 + src1
def : Pat<(v2f64 (fsub VR128:$src1, (fmul VR128:$src2, (memopv2f64 addr:$src3)))),
            (VFNMADDPDr231m VR128:$src1, VR128:$src2, addr:$src3)>;
                        
// FMA231: src1 = - src2*src3 + src1
def : Pat<(v2f64 (fsub VR128:$src1, (fmul VR128:$src2, VR128:$src3))),
            (VFNMADDPDr231r VR128:$src1, VR128:$src2, VR128:$src3)>;
                        
//------------
// FP single precision ADD - 256
//------------

// FMA231: src1 = src2*src3 + src1
def : Pat<(v8f32 (fadd (fmul VR256:$src2, VR256:$src3), VR256:$src1)),
            (VFMADDPSr231rY VR256:$src1, VR256:$src2, VR256:$src3)>;

// FMA213 : src1 = src2*src1 + src3            
def : Pat<(v8f32 (fadd (fmul VR256:$src1, VR256:$src2), (memopv8f32 addr:$src3))),
            (VFMADDPSr213mY VR256:$src1, VR256:$src2, addr:$src3)>;

// FMA231: src1 = src2*src3 + src1
def : Pat<(v8f32 (fadd (fmul (memopv8f32 addr:$src3), VR256:$src2), VR256:$src1)),
            (VFMADDPSr231mY VR256:$src1, VR256:$src2, addr:$src3)>;

// FMA213: src1 = src2*src1 + src3
def : Pat<(v8f32 (fadd (fmul VR256:$src2, VR256:$src1), VR256:$src3)),
            (VFMADDPSr213rY VR256:$src1, VR256:$src2, VR256:$src3)>;

//------------
// FP single precision ADD - 128
//------------

// FMA231 : src1 = src2*src3 + src1            
def : Pat<(v4f32 (fadd (fmul VR128:$src2, (memopv4f32 addr:$src3)), VR128:$src1)),
            (VFMADDPSr231m VR128:$src1, VR128:$src2, addr:$src3)>;

// FMA231 : src1 = src2*src3 + src1            
def : Pat<(v4f32 (fadd (fmul VR128:$src2, VR128:$src3), VR128:$src1)),
        (VFMADDPSr231r VR128:$src1, VR128:$src2, VR128:$src3)>;
            
//------------
// FP single precision SUB - 256
//------------
// FMA231: src1 = src2*src3 - src1
def : Pat<(v8f32 (fsub (fmul VR256:$src2, (memopv8f32 addr:$src3)), VR256:$src1)),
            (VFMSUBPSr231mY VR256:$src1, VR256:$src2, addr:$src3)>;

// FMA231: src1 = src2*src3 - src1
def : Pat<(v8f32 (fsub (fmul VR256:$src2, VR256:$src3), VR256:$src1)),
            (VFMSUBPSr231rY VR256:$src1, VR256:$src2, VR256:$src3)>;

//------------
// FP single precision SUB - 128
//------------
// FMA231 : src1 = src2*src3 - src1            
def : Pat<(v4f32 (fsub (fmul VR128:$src2, (memopv4f32 addr:$src3)), VR128:$src1)),
            (VFMSUBPSr231m VR128:$src1, VR128:$src2, addr:$src3)>;

// FMA231 : src1 = src2*src3 - src1            
def : Pat<(v4f32 (fsub (fmul VR128:$src2, VR128:$src3), VR128:$src1)),
            (VFMSUBPSr231r VR128:$src1, VR128:$src2, VR128:$src3)>;

//------------
// FP single precision FNMADD - 256
//------------
// FMA231: src1 = - src2*src3 + src1
def : Pat<(v8f32 (fsub VR256:$src1, (fmul VR256:$src2, (memopv8f32 addr:$src3)))),
            (VFNMADDPSr231mY VR256:$src1, VR256:$src2, addr:$src3)>;

// FMA231: src1 = - src2*src3 + src1
def : Pat<(v8f32 (fsub VR256:$src1, (fmul VR256:$src2, VR256:$src3))),
            (VFNMADDPSr231rY VR256:$src1, VR256:$src2, VR256:$src3)>;

//------------
// FP single precision FNMADD - 128
//------------

// FMA231 : src1 = src2*src3 - src1            
def : Pat<(v4f32 (fsub VR128:$src1, (fmul VR128:$src2, (memopv4f32 addr:$src3)))),
            (VFNMADDPSr231m VR128:$src1, VR128:$src2, addr:$src3)>;

// FMA231 : src1 = src2*src3 - src1            
def : Pat<(v4f32 (fsub VR128:$src1, (fmul VR128:$src2, VR128:$src3))),
            (VFNMADDPSr231r VR128:$src1, VR128:$src2, VR128:$src3)>;

} // HasFMA3

//------------------------------
// SCALAR
//------------------------------

let Constraints = "$src1 = $dst" in {
multiclass fma3s_rm<bits<8> opc, string OpcodeStr, X86MemOperand x86memop, RegisterClass RC> {
  def r : FMA3<opc, MRMSrcReg, (outs RC:$dst),
           (ins RC:$src1, RC:$src2, RC:$src3),
           !strconcat(OpcodeStr, "\t{$src3, $src2, $dst|$dst, $src2, $src3}"),
           []>;
  def m : FMA3<opc, MRMSrcMem, (outs RC:$dst),
           (ins RC:$src1, RC:$src2, x86memop:$src3),
           !strconcat(OpcodeStr, "\t{$src3, $src2, $dst|$dst, $src2, $src3}"),
           []>;
}

multiclass fma3s_rm_int<bits<8> opc, string OpcodeStr,X86MemOperand x86memop, RegisterClass RC,
           Intrinsic IntId> {
  def r_Int : FMA3<opc, MRMSrcReg, (outs RC:$dst),
           (ins RC:$src1, RC:$src2, RC:$src3),
           !strconcat(OpcodeStr, "\t{$src3, $src2, $dst|$dst, $src2, $src3}"),
           [(set RC:$dst, (IntId RC:$src1, RC:$src3, RC:$src2))]>;
  def m_Int : FMA3<opc, MRMSrcMem, (outs RC:$dst),
           (ins RC:$src1, VR128:$src2, x86memop:$src3),
           !strconcat(OpcodeStr, "\t{$src3, $src2, $dst|$dst, $src2, $src3}"),
           [(set RC:$dst, (IntId RC:$src1, (load addr:$src3), RC:$src2))]>;
}
}

multiclass fma3s_forms<bits<8> opc132, bits<8> opc213, bits<8> opc231,
                       string OpcodeStr, string PackTy, X86MemOperand MemOp, 
                       RegisterClass RC, Intrinsic IntId> {
  defm r132    : fma3s_rm         <opc132, !strconcat(OpcodeStr, !strconcat("132", PackTy)), MemOp, RC>;
  defm r213    : fma3s_rm         <opc213, !strconcat(OpcodeStr, !strconcat("213", PackTy)), MemOp, RC>;
  defm r231    : fma3s_rm         <opc231, !strconcat(OpcodeStr, !strconcat("231", PackTy)), MemOp, RC>;
  defm r132_Int: fma3s_rm_int     <opc132, !strconcat(OpcodeStr, !strconcat("132", PackTy)), MemOp, VR128, IntId>;
}

defm VFMADDSS : fma3s_forms<0x99, 0xA9, 0xB9, "vfmadd", "ss", f32mem, FR32, int_x86_fma4_vfmadd_ss>, VEX_LIG;
defm VFMADDSD : fma3s_forms<0x99, 0xA9, 0xB9, "vfmadd", "sd", f64mem, FR64, int_x86_fma4_vfmadd_sd>, VEX_W, VEX_LIG;
defm VFMSUBSS : fma3s_forms<0x9B, 0xAB, 0xBB, "vfmsub", "ss", f32mem, FR32, int_x86_fma4_vfmsub_ss>, VEX_LIG;
defm VFMSUBSD : fma3s_forms<0x9B, 0xAB, 0xBB, "vfmsub", "sd", f64mem, FR64, int_x86_fma4_vfmsub_sd>, VEX_W, VEX_LIG;

defm VFNMADDSS : fma3s_forms<0x9D, 0xAD, 0xBD, "vfnmadd", "ss", f32mem, FR32, int_x86_fma4_vfnmadd_ss>, VEX_LIG;
defm VFNMADDSD : fma3s_forms<0x9D, 0xAD, 0xBD, "vfnmadd", "sd", f64mem, FR64, int_x86_fma4_vfnmadd_sd>, VEX_W, VEX_LIG;
defm VFNMSUBSS : fma3s_forms<0x9F, 0xAF, 0xBF, "vfnmsub", "ss", f32mem, FR32, int_x86_fma4_vfnmsub_ss>, VEX_LIG;
defm VFNMSUBSD : fma3s_forms<0x9F, 0xAF, 0xBF, "vfnmsub", "sd", f64mem, FR64, int_x86_fma4_vfnmsub_sd>, VEX_W, VEX_LIG;


let Predicates = [HasFMA3], AddedComplexity = 20 in {

//------------
// FP  scalar ADD 
//------------


// FMADD231 : src1 = src2*src3 + src1            
def : Pat<(f32 (fadd (fmul FR32:$src2, FR32:$src3), FR32:$src1)),
            (VFMADDSSr231r FR32:$src1, FR32:$src2, FR32:$src3)>;

def : Pat<(f32 (fadd (fmul FR32:$src2, (loadf32 addr:$src3)), FR32:$src1)),
            (VFMADDSSr231m FR32:$src1, FR32:$src2, addr:$src3)>;

def : Pat<(f64 (fadd (fmul FR64:$src2, FR64:$src3), FR64:$src1)),
            (VFMADDSDr231r FR64:$src1, FR64:$src2, FR64:$src3)>;

def : Pat<(f64 (fadd (fmul FR64:$src2, (loadf64 addr:$src3)), FR64:$src1)),
            (VFMADDSDr231m FR64:$src1, FR64:$src2, addr:$src3)>;



//------------
// FP  scalar SUB src2*src3 - src1
//------------

def : Pat<(f32 (fsub (fmul FR32:$src2, FR32:$src3), FR32:$src1)),
            (VFMSUBSSr231r FR32:$src1, FR32:$src2, FR32:$src3)>;

def : Pat<(f32 (fsub (fmul FR32:$src2, (loadf32 addr:$src3)), FR32:$src1)),
            (VFMSUBSSr231m FR32:$src1, FR32:$src2, addr:$src3)>;

def : Pat<(f64 (fsub (fmul FR64:$src2, FR64:$src3), FR64:$src1)),
            (VFMSUBSDr231r FR64:$src1, FR64:$src2, FR64:$src3)>;

def : Pat<(f64 (fsub (fmul FR64:$src2, (loadf64 addr:$src3)), FR64:$src1)),
            (VFMSUBSDr231m FR64:$src1, FR64:$src2, addr:$src3)>;

//------------
// FP  scalar NADD src1 - src2*src3
//------------

def : Pat<(f32 (fsub FR32:$src1, (fmul FR32:$src2, FR32:$src3))),
            (VFNMADDSSr231r FR32:$src1, FR32:$src2, FR32:$src3)>;

def : Pat<(f32 (fsub FR32:$src1, (fmul FR32:$src2, (loadf32 addr:$src3)))),
            (VFNMADDSSr231m FR32:$src1, FR32:$src2, addr:$src3)>;

def : Pat<(f64 (fsub FR64:$src1, (fmul FR64:$src2, FR64:$src3))),
            (VFNMADDSDr231r FR64:$src1, FR64:$src2, FR64:$src3)>;

def : Pat<(f64 (fsub FR64:$src1, (fmul FR64:$src2, (loadf64 addr:$src3)))),
            (VFNMADDSDr231m FR64:$src1, FR64:$src2, addr:$src3)>;

} // HasFMA3

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
                          int_x86_fma4_vfmadd_ss>;
defm VFMADDSD4    : fma4s<0x6B, "vfmaddsd", sdmem, sse_load_f64,
                          int_x86_fma4_vfmadd_sd>;
defm VFMADDPS4    : fma4p<0x68, "vfmaddps", int_x86_fma4_vfmadd_ps,
                          int_x86_fma4_vfmadd_ps_256, memopv4f32, memopv8f32>;
defm VFMADDPD4    : fma4p<0x69, "vfmaddpd", int_x86_fma4_vfmadd_pd,
                          int_x86_fma4_vfmadd_pd_256, memopv2f64, memopv4f64>;
defm VFMSUBSS4    : fma4s<0x6E, "vfmsubss", ssmem, sse_load_f32,
                          int_x86_fma4_vfmsub_ss>;
defm VFMSUBSD4    : fma4s<0x6F, "vfmsubsd", sdmem, sse_load_f64,
                          int_x86_fma4_vfmsub_sd>;
defm VFMSUBPS4    : fma4p<0x6C, "vfmsubps", int_x86_fma4_vfmsub_ps,
                          int_x86_fma4_vfmsub_ps_256, memopv4f32, memopv8f32>;
defm VFMSUBPD4    : fma4p<0x6D, "vfmsubpd", int_x86_fma4_vfmsub_pd,
                          int_x86_fma4_vfmsub_pd_256, memopv2f64, memopv4f64>;
defm VFNMADDSS4   : fma4s<0x7A, "vfnmaddss", ssmem, sse_load_f32,
                          int_x86_fma4_vfnmadd_ss>;
defm VFNMADDSD4   : fma4s<0x7B, "vfnmaddsd", sdmem, sse_load_f64,
                          int_x86_fma4_vfnmadd_sd>;
defm VFNMADDPS4   : fma4p<0x78, "vfnmaddps", int_x86_fma4_vfnmadd_ps,
                          int_x86_fma4_vfnmadd_ps_256, memopv4f32, memopv8f32>;
defm VFNMADDPD4   : fma4p<0x79, "vfnmaddpd", int_x86_fma4_vfnmadd_pd,
                          int_x86_fma4_vfnmadd_pd_256, memopv2f64, memopv4f64>;
defm VFNMSUBSS4   : fma4s<0x7E, "vfnmsubss", ssmem, sse_load_f32,
                          int_x86_fma4_vfnmsub_ss>;
defm VFNMSUBSD4   : fma4s<0x7F, "vfnmsubsd", sdmem, sse_load_f64,
                          int_x86_fma4_vfnmsub_sd>;
defm VFNMSUBPS4   : fma4p<0x7C, "vfnmsubps", int_x86_fma4_vfnmsub_ps,
                          int_x86_fma4_vfnmsub_ps_256, memopv4f32, memopv8f32>;
defm VFNMSUBPD4   : fma4p<0x7D, "vfnmsubpd", int_x86_fma4_vfnmsub_pd,
                          int_x86_fma4_vfnmsub_pd_256, memopv2f64, memopv4f64>;
defm VFMADDSUBPS4 : fma4p<0x5C, "vfmaddsubps", int_x86_fma4_vfmaddsub_ps,
                         int_x86_fma4_vfmaddsub_ps_256, memopv4f32, memopv8f32>;
defm VFMADDSUBPD4 : fma4p<0x5D, "vfmaddsubpd", int_x86_fma4_vfmaddsub_pd,
                         int_x86_fma4_vfmaddsub_pd_256, memopv2f64, memopv4f64>;
defm VFMSUBADDPS4 : fma4p<0x5E, "vfmsubaddps", int_x86_fma4_vfmsubadd_ps,
                         int_x86_fma4_vfmsubadd_ps_256, memopv4f32, memopv8f32>;
defm VFMSUBADDPD4 : fma4p<0x5F, "vfmsubaddpd", int_x86_fma4_vfmsubadd_pd,
                         int_x86_fma4_vfmsubadd_pd_256, memopv2f64, memopv4f64>;
} // HasFMA4

