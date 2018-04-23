//===-- X86InstrMovdir.td - Direct Move Instructions ------*- tablegen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// MOVDIRI - Move doubleword/quadword as direct store
def MOVDIRI32 : I<0xF9, MRMDestMem, (outs), (ins i32mem:$dst, GR32:$src),
                  "movdiri\t{$src, $dst|$dst, $src}",
                  [(int_x86_directstore32 addr:$dst, GR32:$src)]>,
                 T8, Requires<[HasMOVDIRI]>;
def MOVDIRI64 : RI<0xF9, MRMDestMem, (outs), (ins i64mem:$dst, GR64:$src),
                   "movdiri\t{$src, $dst|$dst, $src}",
                   [(int_x86_directstore64 addr:$dst, GR64:$src)]>,
                  T8, Requires<[In64BitMode, HasMOVDIRI]>;

// MOVDIR64B - Move 64 bytes as direct store
def MOVDIR64B16 : I<0xF8, MRMSrcMem, (outs), (ins GR16:$dst, i512mem:$src),
                    "movdir64b\t{$src, $dst|$dst, $src}", []>,
                   T8PD, AdSize16, Requires<[HasMOVDIR64B, Not64BitMode]>;
def MOVDIR64B32 : I<0xF8, MRMSrcMem, (outs), (ins GR32:$dst, i512mem:$src),
                    "movdir64b\t{$src, $dst|$dst, $src}",
                    [(int_x86_movdir64b GR32:$dst, addr:$src)]>,
                   T8PD, AdSize32, Requires<[HasMOVDIR64B]>;
def MOVDIR64B64 : I<0xF8, MRMSrcMem, (outs), (ins GR64:$dst, i512mem:$src),
                    "movdir64b\t{$src, $dst|$dst, $src}",
                    [(int_x86_movdir64b GR64:$dst, addr:$src)]>,
                   T8PD, AdSize64, Requires<[HasMOVDIR64B, In64BitMode]>;
