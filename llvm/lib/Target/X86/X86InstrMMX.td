//====- X86InstrMMX.td - Describe the X86 Instruction Set -------*- C++ -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the Evan Cheng and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file describes the X86 MMX instruction set, defining the instructions,
// and properties of the instructions which are needed for code generation,
// machine code emission, and analysis.
//
//===----------------------------------------------------------------------===//

// Move Instructions
def MOVD64rr : I<0x6E, MRMSrcReg, (ops VR64:$dst, R32:$src),
                 "movd {$src, $dst|$dst, $src}", []>, TB,
               Requires<[HasMMX]>;
def MOVD64rm : I<0x6E, MRMSrcMem, (ops VR64:$dst, i32mem:$src),
                 "movd {$src, $dst|$dst, $src}", []>, TB,
               Requires<[HasMMX]>;
def MOVD64mr : I<0x7E, MRMDestMem, (ops i32mem:$dst, VR64:$src),
                 "movd {$src, $dst|$dst, $src}", []>, TB,
               Requires<[HasMMX]>;

def MOVD128rr : I<0x6E, MRMSrcReg, (ops VR128:$dst, R32:$src),
                  "movd {$src, $dst|$dst, $src}", []>, TB, OpSize,
                Requires<[HasSSE2]>;
def MOVD128rm : I<0x6E, MRMSrcMem, (ops VR128:$dst, i32mem:$src),
                  "movd {$src, $dst|$dst, $src}", []>, TB, OpSize,
                Requires<[HasSSE2]>;
def MOVD128mr : I<0x7E, MRMDestMem, (ops i32mem:$dst, VR128:$src),
                  "movd {$src, $dst|$dst, $src}", []>, TB, OpSize,
                Requires<[HasSSE2]>;


def MOVQ64rr : I<0x6F, MRMSrcReg, (ops VR64:$dst, VR64:$src),
                 "movq {$src, $dst|$dst, $src}", []>, TB,
               Requires<[HasMMX]>;
def MOVQ64rm : I<0x6F, MRMSrcMem, (ops VR64:$dst, i64mem:$src),
                 "movq {$src, $dst|$dst, $src}", []>, TB,
               Requires<[HasMMX]>;
def MOVQ64mr : I<0x7F, MRMDestMem, (ops i64mem:$dst, VR64:$src),
                 "movq {$src, $dst|$dst, $src}", []>, TB,
               Requires<[HasMMX]>;

def MOVQ128rr : I<0x7E, MRMSrcReg, (ops VR128:$dst, VR64:$src),
                  "movq {$src, $dst|$dst, $src}", []>, XS,
                Requires<[HasSSE2]>;
def MOVQ128rm : I<0x7E, MRMSrcMem, (ops VR128:$dst, i64mem:$src),
                  "movq {$src, $dst|$dst, $src}", []>, XS,
                Requires<[HasSSE2]>;
def MOVQ128mr : I<0xD6, MRMSrcMem, (ops i64mem:$dst, VR128:$src),
                  "movq {$src, $dst|$dst, $src}", []>, TB, OpSize,
                Requires<[HasSSE2]>;

