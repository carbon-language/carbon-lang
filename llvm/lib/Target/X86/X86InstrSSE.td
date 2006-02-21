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

//===----------------------------------------------------------------------===//
// XMM Floating point support (requires SSE / SSE2)
//===----------------------------------------------------------------------===//

def MOVSSrr : I<0x10, MRMSrcReg, (ops FR32:$dst, FR32:$src),
                "movss {$src, $dst|$dst, $src}", []>,
              Requires<[HasSSE1]>, XS;
def MOVSDrr : I<0x10, MRMSrcReg, (ops FR64:$dst, FR64:$src),
                "movsd {$src, $dst|$dst, $src}", []>,
              Requires<[HasSSE2]>, XD;

def MOVSSrm : I<0x10, MRMSrcMem, (ops FR32:$dst, f32mem:$src),
                "movss {$src, $dst|$dst, $src}",
                [(set FR32:$dst, (loadf32 addr:$src))]>,
              Requires<[HasSSE1]>, XS;
def MOVSSmr : I<0x11, MRMDestMem, (ops f32mem:$dst, FR32:$src),
                "movss {$src, $dst|$dst, $src}",
                [(store FR32:$src, addr:$dst)]>,
              Requires<[HasSSE1]>, XS;
def MOVSDrm : I<0x10, MRMSrcMem, (ops FR64:$dst, f64mem:$src),
                "movsd {$src, $dst|$dst, $src}",
                [(set FR64:$dst, (loadf64 addr:$src))]>,
              Requires<[HasSSE2]>, XD;
def MOVSDmr : I<0x11, MRMDestMem, (ops f64mem:$dst, FR64:$src),
                "movsd {$src, $dst|$dst, $src}",
                [(store FR64:$src, addr:$dst)]>,
              Requires<[HasSSE2]>, XD;

def CVTTSS2SIrr: I<0x2C, MRMSrcReg, (ops R32:$dst, FR32:$src),
                   "cvttss2si {$src, $dst|$dst, $src}",
                   [(set R32:$dst, (fp_to_sint FR32:$src))]>,
                 Requires<[HasSSE1]>, XS;
def CVTTSS2SIrm: I<0x2C, MRMSrcMem, (ops R32:$dst, f32mem:$src),
                   "cvttss2si {$src, $dst|$dst, $src}",
                   [(set R32:$dst, (fp_to_sint (loadf32 addr:$src)))]>,
                 Requires<[HasSSE1]>, XS;
def CVTTSD2SIrr: I<0x2C, MRMSrcReg, (ops R32:$dst, FR64:$src),
                   "cvttsd2si {$src, $dst|$dst, $src}",
                   [(set R32:$dst, (fp_to_sint FR64:$src))]>,
                 Requires<[HasSSE2]>, XD;
def CVTTSD2SIrm: I<0x2C, MRMSrcMem, (ops R32:$dst, f64mem:$src),
                   "cvttsd2si {$src, $dst|$dst, $src}",
                   [(set R32:$dst, (fp_to_sint (loadf64 addr:$src)))]>,
                 Requires<[HasSSE2]>, XD;
def CVTSS2SDrr: I<0x5A, MRMSrcReg, (ops FR64:$dst, FR32:$src),
                  "cvtss2sd {$src, $dst|$dst, $src}",
                  [(set FR64:$dst, (fextend FR32:$src))]>,
                Requires<[HasSSE2]>, XS;
def CVTSS2SDrm: I<0x5A, MRMSrcMem, (ops FR64:$dst, f32mem:$src),
                  "cvtss2sd {$src, $dst|$dst, $src}",
                  [(set FR64:$dst, (fextend (loadf32 addr:$src)))]>,
                Requires<[HasSSE2]>, XS;
def CVTSD2SSrr: I<0x5A, MRMSrcReg, (ops FR32:$dst, FR64:$src),
                  "cvtsd2ss {$src, $dst|$dst, $src}",
                  [(set FR32:$dst, (fround FR64:$src))]>,
                Requires<[HasSSE2]>, XD;
def CVTSD2SSrm: I<0x5A, MRMSrcMem, (ops FR32:$dst, f64mem:$src), 
                  "cvtsd2ss {$src, $dst|$dst, $src}",
                  [(set FR32:$dst, (fround (loadf64 addr:$src)))]>,
                Requires<[HasSSE2]>, XD;
def CVTSI2SSrr: I<0x2A, MRMSrcReg, (ops FR32:$dst, R32:$src),
                  "cvtsi2ss {$src, $dst|$dst, $src}",
                  [(set FR32:$dst, (sint_to_fp R32:$src))]>,
                Requires<[HasSSE2]>, XS;
def CVTSI2SSrm: I<0x2A, MRMSrcMem, (ops FR32:$dst, i32mem:$src),
                  "cvtsi2ss {$src, $dst|$dst, $src}",
                  [(set FR32:$dst, (sint_to_fp (loadi32 addr:$src)))]>,
                Requires<[HasSSE2]>, XS;
def CVTSI2SDrr: I<0x2A, MRMSrcReg, (ops FR64:$dst, R32:$src),
                  "cvtsi2sd {$src, $dst|$dst, $src}",
                  [(set FR64:$dst, (sint_to_fp R32:$src))]>,
                Requires<[HasSSE2]>, XD;
def CVTSI2SDrm: I<0x2A, MRMSrcMem, (ops FR64:$dst, i32mem:$src),
                  "cvtsi2sd {$src, $dst|$dst, $src}",
                  [(set FR64:$dst, (sint_to_fp (loadi32 addr:$src)))]>,
                Requires<[HasSSE2]>, XD;

def SQRTSSrr : I<0x51, MRMSrcReg, (ops FR32:$dst, FR32:$src),
                 "sqrtss {$src, $dst|$dst, $src}",
                 [(set FR32:$dst, (fsqrt FR32:$src))]>,
               Requires<[HasSSE1]>, XS;
def SQRTSSrm : I<0x51, MRMSrcMem, (ops FR32:$dst, f32mem:$src),
                 "sqrtss {$src, $dst|$dst, $src}",
                 [(set FR32:$dst, (fsqrt (loadf32 addr:$src)))]>,
               Requires<[HasSSE1]>, XS;
def SQRTSDrr : I<0x51, MRMSrcReg, (ops FR64:$dst, FR64:$src),
                 "sqrtsd {$src, $dst|$dst, $src}",
                 [(set FR64:$dst, (fsqrt FR64:$src))]>,
               Requires<[HasSSE2]>, XD;
def SQRTSDrm : I<0x51, MRMSrcMem, (ops FR64:$dst, f64mem:$src),
                 "sqrtsd {$src, $dst|$dst, $src}",
                 [(set FR64:$dst, (fsqrt (loadf64 addr:$src)))]>,
               Requires<[HasSSE2]>, XD;

def UCOMISSrr: I<0x2E, MRMSrcReg, (ops FR32:$src1, FR32:$src2),
                 "ucomiss {$src2, $src1|$src1, $src2}",
                 [(X86cmp FR32:$src1, FR32:$src2)]>,
               Requires<[HasSSE1]>, TB;
def UCOMISSrm: I<0x2E, MRMSrcMem, (ops FR32:$src1, f32mem:$src2),
                 "ucomiss {$src2, $src1|$src1, $src2}",
                 [(X86cmp FR32:$src1, (loadf32 addr:$src2))]>,
               Requires<[HasSSE1]>, TB;
def UCOMISDrr: I<0x2E, MRMSrcReg, (ops FR64:$src1, FR64:$src2),
                 "ucomisd {$src2, $src1|$src1, $src2}",
                 [(X86cmp FR64:$src1, FR64:$src2)]>,
               Requires<[HasSSE2]>, TB, OpSize;
def UCOMISDrm: I<0x2E, MRMSrcMem, (ops FR64:$src1, f64mem:$src2),
                 "ucomisd {$src2, $src1|$src1, $src2}",
                 [(X86cmp FR64:$src1, (loadf64 addr:$src2))]>,
               Requires<[HasSSE2]>, TB, OpSize;

let isTwoAddress = 1 in {
// SSE Scalar Arithmetic
let isCommutable = 1 in {
def ADDSSrr : I<0x58, MRMSrcReg, (ops FR32:$dst, FR32:$src1, FR32:$src2),
                "addss {$src2, $dst|$dst, $src2}",
                [(set FR32:$dst, (fadd FR32:$src1, FR32:$src2))]>,
              Requires<[HasSSE1]>, XS;
def ADDSDrr : I<0x58, MRMSrcReg, (ops FR64:$dst, FR64:$src1, FR64:$src2),
                "addsd {$src2, $dst|$dst, $src2}",
                [(set FR64:$dst, (fadd FR64:$src1, FR64:$src2))]>,
              Requires<[HasSSE2]>, XD;
def MULSSrr : I<0x59, MRMSrcReg, (ops FR32:$dst, FR32:$src1, FR32:$src2),
                "mulss {$src2, $dst|$dst, $src2}",
                [(set FR32:$dst, (fmul FR32:$src1, FR32:$src2))]>,
              Requires<[HasSSE1]>, XS;
def MULSDrr : I<0x59, MRMSrcReg, (ops FR64:$dst, FR64:$src1, FR64:$src2),
                "mulsd {$src2, $dst|$dst, $src2}",
                [(set FR64:$dst, (fmul FR64:$src1, FR64:$src2))]>,
              Requires<[HasSSE2]>, XD;
}

def ADDSSrm : I<0x58, MRMSrcMem, (ops FR32:$dst, FR32:$src1, f32mem:$src2),
                "addss {$src2, $dst|$dst, $src2}",
                [(set FR32:$dst, (fadd FR32:$src1, (loadf32 addr:$src2)))]>,
              Requires<[HasSSE1]>, XS;
def ADDSDrm : I<0x58, MRMSrcMem, (ops FR64:$dst, FR64:$src1, f64mem:$src2),
                "addsd {$src2, $dst|$dst, $src2}",
                [(set FR64:$dst, (fadd FR64:$src1, (loadf64 addr:$src2)))]>,
              Requires<[HasSSE2]>, XD;
def MULSSrm : I<0x59, MRMSrcMem, (ops FR32:$dst, FR32:$src1, f32mem:$src2),
                "mulss {$src2, $dst|$dst, $src2}",
                [(set FR32:$dst, (fmul FR32:$src1, (loadf32 addr:$src2)))]>,
              Requires<[HasSSE1]>, XS;
def MULSDrm : I<0x59, MRMSrcMem, (ops FR64:$dst, FR64:$src1, f64mem:$src2),
                "mulsd {$src2, $dst|$dst, $src2}",
                [(set FR64:$dst, (fmul FR64:$src1, (loadf64 addr:$src2)))]>,
              Requires<[HasSSE2]>, XD;

def DIVSSrr : I<0x5E, MRMSrcReg, (ops FR32:$dst, FR32:$src1, FR32:$src2),
                "divss {$src2, $dst|$dst, $src2}",
                [(set FR32:$dst, (fdiv FR32:$src1, FR32:$src2))]>,
              Requires<[HasSSE1]>, XS;
def DIVSSrm : I<0x5E, MRMSrcMem, (ops FR32:$dst, FR32:$src1, f32mem:$src2),
                "divss {$src2, $dst|$dst, $src2}",
                [(set FR32:$dst, (fdiv FR32:$src1, (loadf32 addr:$src2)))]>,
              Requires<[HasSSE1]>, XS;
def DIVSDrr : I<0x5E, MRMSrcReg, (ops FR64:$dst, FR64:$src1, FR64:$src2),
                "divsd {$src2, $dst|$dst, $src2}",
                [(set FR64:$dst, (fdiv FR64:$src1, FR64:$src2))]>,
              Requires<[HasSSE2]>, XD;
def DIVSDrm : I<0x5E, MRMSrcMem, (ops FR64:$dst, FR64:$src1, f64mem:$src2),
                "divsd {$src2, $dst|$dst, $src2}",
                [(set FR64:$dst, (fdiv FR64:$src1, (loadf64 addr:$src2)))]>,
              Requires<[HasSSE2]>, XD;

def SUBSSrr : I<0x5C, MRMSrcReg, (ops FR32:$dst, FR32:$src1, FR32:$src2),
                "subss {$src2, $dst|$dst, $src2}",
                [(set FR32:$dst, (fsub FR32:$src1, FR32:$src2))]>,
              Requires<[HasSSE1]>, XS;
def SUBSSrm : I<0x5C, MRMSrcMem, (ops FR32:$dst, FR32:$src1, f32mem:$src2),
                "subss {$src2, $dst|$dst, $src2}",
                [(set FR32:$dst, (fsub FR32:$src1, (loadf32 addr:$src2)))]>,
              Requires<[HasSSE1]>, XS;
def SUBSDrr : I<0x5C, MRMSrcReg, (ops FR64:$dst, FR64:$src1, FR64:$src2),
                "subsd {$src2, $dst|$dst, $src2}",
                [(set FR64:$dst, (fsub FR64:$src1, FR64:$src2))]>,
              Requires<[HasSSE2]>, XD;
def SUBSDrm : I<0x5C, MRMSrcMem, (ops FR64:$dst, FR64:$src1, f64mem:$src2),
                "subsd {$src2, $dst|$dst, $src2}",
                [(set FR64:$dst, (fsub FR64:$src1, (loadf64 addr:$src2)))]>,
              Requires<[HasSSE2]>, XD;

// SSE compare
def CMPSSrr : I<0xC2, MRMSrcReg, 
                (ops FR32:$dst, FR32:$src1, FR32:$src, SSECC:$cc),
                "cmp${cc}ss {$src, $dst|$dst, $src}", []>,
              Requires<[HasSSE1]>, XS;
def CMPSSrm : I<0xC2, MRMSrcMem, 
                (ops FR32:$dst, FR32:$src1, f32mem:$src, SSECC:$cc),
                "cmp${cc}ss {$src, $dst|$dst, $src}", []>,
              Requires<[HasSSE1]>, XS;
def CMPSDrr : I<0xC2, MRMSrcReg, 
                (ops FR64:$dst, FR64:$src1, FR64:$src, SSECC:$cc),
                "cmp${cc}sd {$src, $dst|$dst, $src}", []>,
              Requires<[HasSSE1]>, XD;
def CMPSDrm : I<0xC2, MRMSrcMem, 
                (ops FR64:$dst, FR64:$src1, f64mem:$src, SSECC:$cc),
                "cmp${cc}sd {$src, $dst|$dst, $src}", []>,
              Requires<[HasSSE2]>, XD;
}


//===----------------------------------------------------------------------===//
// Alias Instructions
//===----------------------------------------------------------------------===//

// Alias instructions that map fld0 to pxor for sse.
// FIXME: remove when we can teach regalloc that xor reg, reg is ok.
def FsFLD0SS : I<0xEF, MRMInitReg, (ops FR32:$dst),
                 "pxor $dst, $dst", [(set FR32:$dst, fp32imm0)]>,
               Requires<[HasSSE1]>, TB, OpSize;
def FsFLD0SD : I<0xEF, MRMInitReg, (ops FR64:$dst),
                 "pxor $dst, $dst", [(set FR64:$dst, fp64imm0)]>,
               Requires<[HasSSE2]>, TB, OpSize;

// Alias instructions to do FR32 / FR64 reg-to-reg copy using movaps / movapd.
// Upper bits are disregarded.
def FsMOVAPSrr : I<0x28, MRMSrcReg, (ops V4F32:$dst, V4F32:$src),
                   "movaps {$src, $dst|$dst, $src}", []>,
                 Requires<[HasSSE1]>, TB;
def FsMOVAPDrr : I<0x28, MRMSrcReg, (ops V2F64:$dst, V2F64:$src),
                   "movapd {$src, $dst|$dst, $src}", []>,
                 Requires<[HasSSE2]>, TB, OpSize;

// Alias instructions to load FR32 / FR64 from f128mem using movaps / movapd.
// Upper bits are disregarded.
def FsMOVAPSrm : I<0x28, MRMSrcMem, (ops FR32:$dst, f128mem:$src),
                   "movaps {$src, $dst|$dst, $src}",
                   [(set FR32:$dst, (X86loadpf32 addr:$src))]>,
                 Requires<[HasSSE1]>, TB;
def FsMOVAPDrm : I<0x28, MRMSrcMem, (ops FR64:$dst, f128mem:$src),
                  "movapd {$src, $dst|$dst, $src}",
                  [(set FR64:$dst, (X86loadpf64 addr:$src))]>,
                Requires<[HasSSE2]>, TB, OpSize;

// Alias bitwise logical operations using SSE logical ops on packed FP values.
let isTwoAddress = 1 in {
let isCommutable = 1 in {
def FsANDPSrr : I<0x54, MRMSrcReg, (ops FR32:$dst, FR32:$src1, FR32:$src2),
                  "andps {$src2, $dst|$dst, $src2}",
                  [(set FR32:$dst, (X86fand FR32:$src1, FR32:$src2))]>,
                Requires<[HasSSE1]>, TB;
def FsANDPDrr : I<0x54, MRMSrcReg, (ops FR64:$dst, FR64:$src1, FR64:$src2),
                  "andpd {$src2, $dst|$dst, $src2}",
                  [(set FR64:$dst, (X86fand FR64:$src1, FR64:$src2))]>,
                Requires<[HasSSE2]>, TB, OpSize;
def FsORPSrr : I<0x56, MRMSrcReg, (ops FR32:$dst, FR32:$src1, FR32:$src2),
                 "orps {$src2, $dst|$dst, $src2}", []>,
               Requires<[HasSSE1]>, TB;
def FsORPDrr : I<0x56, MRMSrcReg, (ops FR64:$dst, FR64:$src1, FR64:$src2),
                 "orpd {$src2, $dst|$dst, $src2}", []>,
               Requires<[HasSSE2]>, TB, OpSize;
def FsXORPSrr : I<0x57, MRMSrcReg, (ops FR32:$dst, FR32:$src1, FR32:$src2),
                  "xorps {$src2, $dst|$dst, $src2}",
                  [(set FR32:$dst, (X86fxor FR32:$src1, FR32:$src2))]>,
                Requires<[HasSSE1]>, TB;
def FsXORPDrr : I<0x57, MRMSrcReg, (ops FR64:$dst, FR64:$src1, FR64:$src2),
                  "xorpd {$src2, $dst|$dst, $src2}",
                  [(set FR64:$dst, (X86fxor FR64:$src1, FR64:$src2))]>,
                Requires<[HasSSE2]>, TB, OpSize;
}
def FsANDPSrm : I<0x54, MRMSrcMem, (ops FR32:$dst, FR32:$src1, f128mem:$src2),
                  "andps {$src2, $dst|$dst, $src2}",
                  [(set FR32:$dst, (X86fand FR32:$src1,
                                    (X86loadpf32 addr:$src2)))]>,
                Requires<[HasSSE1]>, TB;
def FsANDPDrm : I<0x54, MRMSrcMem, (ops FR64:$dst, FR64:$src1, f128mem:$src2),
                  "andpd {$src2, $dst|$dst, $src2}",
                  [(set FR64:$dst, (X86fand FR64:$src1,
                                    (X86loadpf64 addr:$src2)))]>,
                Requires<[HasSSE2]>, TB, OpSize;
def FsORPSrm : I<0x56, MRMSrcMem, (ops FR32:$dst, FR32:$src1, f128mem:$src2),
                 "orps {$src2, $dst|$dst, $src2}", []>,
               Requires<[HasSSE1]>, TB;
def FsORPDrm : I<0x56, MRMSrcMem, (ops FR64:$dst, FR64:$src1, f128mem:$src2),
                 "orpd {$src2, $dst|$dst, $src2}", []>,
               Requires<[HasSSE2]>, TB, OpSize;
def FsXORPSrm : I<0x57, MRMSrcMem, (ops FR32:$dst, FR32:$src1, f128mem:$src2),
                  "xorps {$src2, $dst|$dst, $src2}",
                  [(set FR32:$dst, (X86fxor FR32:$src1,
                                    (X86loadpf32 addr:$src2)))]>,
                Requires<[HasSSE1]>, TB;
def FsXORPDrm : I<0x57, MRMSrcMem, (ops FR64:$dst, FR64:$src1, f128mem:$src2),
                  "xorpd {$src2, $dst|$dst, $src2}",
                  [(set FR64:$dst, (X86fxor FR64:$src1,
                                    (X86loadpf64 addr:$src2)))]>,
                Requires<[HasSSE2]>, TB, OpSize;

def FsANDNPSrr : I<0x55, MRMSrcReg, (ops FR32:$dst, FR32:$src1, FR32:$src2),
                   "andnps {$src2, $dst|$dst, $src2}", []>,
                 Requires<[HasSSE1]>, TB;
def FsANDNPSrm : I<0x55, MRMSrcMem, (ops FR32:$dst, FR32:$src1, f128mem:$src2),
                   "andnps {$src2, $dst|$dst, $src2}", []>,
                 Requires<[HasSSE1]>, TB;
def FsANDNPDrr : I<0x55, MRMSrcReg, (ops FR64:$dst, FR64:$src1, FR64:$src2),
                   "andnpd {$src2, $dst|$dst, $src2}", []>,
                 Requires<[HasSSE2]>, TB, OpSize;
def FsANDNPDrm : I<0x55, MRMSrcMem, (ops FR64:$dst, FR64:$src1, f128mem:$src2),
                   "andnpd {$src2, $dst|$dst, $src2}", []>,
                 Requires<[HasSSE2]>, TB, OpSize;
}
