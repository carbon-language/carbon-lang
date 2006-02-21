//====- X86InstrSSE.td - Describe the X86 Instruction Set -------*- C++ -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the Evan Cheng and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file describes the X86 SSE instruction set, defining the instructions,
// and properties of the instructions which are needed for code generation,
// machine code emission, and analysis.
//
//===----------------------------------------------------------------------===//

def MOVAPSrr : I<0x28, MRMSrcReg, (ops V4F32:$dst, V4F32:$src),
                "movaps {$src, $dst|$dst, $src}", []>,
               Requires<[HasSSE1]>, TB;
def MOVAPDrr : I<0x28, MRMSrcReg, (ops V2F64:$dst, V2F64:$src),
                "movapd {$src, $dst|$dst, $src}", []>,
               Requires<[HasSSE2]>, TB, OpSize;

def MOVAPSrm : I<0x28, MRMSrcMem, (ops V4F32:$dst, f128mem:$src),
                "movaps {$src, $dst|$dst, $src}", []>,
               Requires<[HasSSE1]>, TB;
def MOVAPSmr : I<0x29, MRMDestMem, (ops f128mem:$dst, V4F32:$src),
                "movaps {$src, $dst|$dst, $src}",[]>,
               Requires<[HasSSE1]>, TB;
def MOVAPDrm : I<0x28, MRMSrcMem, (ops V2F64:$dst, f128mem:$src),
                "movapd {$src, $dst|$dst, $src}", []>,
               Requires<[HasSSE1]>, TB, OpSize;
def MOVAPDmr : I<0x29, MRMDestMem, (ops f128mem:$dst, V2F64:$src),
                "movapd {$src, $dst|$dst, $src}",[]>,
               Requires<[HasSSE2]>, TB, OpSize;

// Logical
let isTwoAddress = 1 in {
let isCommutable = 1 in {
def ANDPSrr : I<0x54, MRMSrcReg, (ops V4F32:$dst, V4F32:$src1, V4F32:$src2),
                "andps {$src2, $dst|$dst, $src2}",
                [(set V4F32:$dst, (X86fand V4F32:$src1, V4F32:$src2))]>,
              Requires<[HasSSE1]>, TB;
def ANDPDrr : I<0x54, MRMSrcReg, (ops V2F64:$dst, V2F64:$src1, V2F64:$src2),
                "andpd {$src2, $dst|$dst, $src2}",
                [(set V2F64:$dst, (X86fand V2F64:$src1, V2F64:$src2))]>,
              Requires<[HasSSE2]>, TB, OpSize;
def ORPSrr : I<0x56, MRMSrcReg, (ops V4F32:$dst, V4F32:$src1, V4F32:$src2),
                "orps {$src2, $dst|$dst, $src2}", []>,
             Requires<[HasSSE1]>, TB;
def ORPDrr : I<0x56, MRMSrcReg, (ops V2F64:$dst, V2F64:$src1, V2F64:$src2),
                "orpd {$src2, $dst|$dst, $src2}", []>,
             Requires<[HasSSE2]>, TB, OpSize;
def XORPSrr : I<0x57, MRMSrcReg, (ops V4F32:$dst, V4F32:$src1, V4F32:$src2),
                "xorps {$src2, $dst|$dst, $src2}",
                [(set V4F32:$dst, (X86fxor V4F32:$src1, V4F32:$src2))]>,
              Requires<[HasSSE1]>, TB;
def XORPDrr : I<0x57, MRMSrcReg, (ops V2F64:$dst, V2F64:$src1, V2F64:$src2),
                "xorpd {$src2, $dst|$dst, $src2}",
                [(set V2F64:$dst, (X86fxor V2F64:$src1, V2F64:$src2))]>,
              Requires<[HasSSE2]>, TB, OpSize;
}
def ANDPSrm : I<0x54, MRMSrcMem, (ops V4F32:$dst, V4F32:$src1, f128mem:$src2),
                "andps {$src2, $dst|$dst, $src2}",
                [(set V4F32:$dst, (X86fand V4F32:$src1,
                                  (X86loadpv4f32 addr:$src2)))]>,
              Requires<[HasSSE1]>, TB;
def ANDPDrm : I<0x54, MRMSrcMem, (ops V2F64:$dst, V2F64:$src1, f128mem:$src2),
                "andpd {$src2, $dst|$dst, $src2}",
                [(set V2F64:$dst, (X86fand V2F64:$src1,
                                  (X86loadpv2f64 addr:$src2)))]>,
              Requires<[HasSSE2]>, TB, OpSize;
def ORPSrm : I<0x56, MRMSrcMem, (ops V4F32:$dst, V4F32:$src1, f128mem:$src2),
                "orps {$src2, $dst|$dst, $src2}", []>,
             Requires<[HasSSE1]>, TB;
def ORPDrm : I<0x56, MRMSrcMem, (ops V2F64:$dst, V2F64:$src1, f128mem:$src2),
                "orpd {$src2, $dst|$dst, $src2}", []>,
             Requires<[HasSSE2]>, TB, OpSize;
def XORPSrm : I<0x57, MRMSrcMem, (ops V4F32:$dst, V4F32:$src1, f128mem:$src2),
                "xorps {$src2, $dst|$dst, $src2}",
                [(set V4F32:$dst, (X86fxor V4F32:$src1,
                                  (X86loadpv4f32 addr:$src2)))]>,
              Requires<[HasSSE1]>, TB;
def XORPDrm : I<0x57, MRMSrcMem, (ops V2F64:$dst, V2F64:$src1, f128mem:$src2),
                "xorpd {$src2, $dst|$dst, $src2}",
                [(set V2F64:$dst, (X86fxor V2F64:$src1,
                                  (X86loadpv2f64 addr:$src2)))]>,
              Requires<[HasSSE2]>, TB, OpSize;

def ANDNPSrr : I<0x55, MRMSrcReg, (ops V4F32:$dst, V4F32:$src1, V4F32:$src2),
                "andnps {$src2, $dst|$dst, $src2}", []>,
               Requires<[HasSSE1]>, TB;
def ANDNPSrm : I<0x55, MRMSrcMem, (ops V4F32:$dst, V4F32:$src1, f128mem:$src2),
                "andnps {$src2, $dst|$dst, $src2}", []>,
               Requires<[HasSSE1]>, TB;
def ANDNPDrr : I<0x55, MRMSrcReg, (ops V2F64:$dst, V2F64:$src1, V2F64:$src2),
                "andnpd {$src2, $dst|$dst, $src2}", []>,
               Requires<[HasSSE2]>, TB, OpSize;
def ANDNPDrm : I<0x55, MRMSrcMem, (ops V2F64:$dst, V2F64:$src1, f128mem:$src2),
                "andnpd {$src2, $dst|$dst, $src2}", []>,
               Requires<[HasSSE2]>, TB, OpSize;
}
