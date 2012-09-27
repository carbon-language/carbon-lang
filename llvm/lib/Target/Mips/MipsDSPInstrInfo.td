//===- MipsDSPInstrInfo.td - DSP ASE instructions -*- tablegen ------------*-=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file describes Mips DSP ASE instructions.
//
//===----------------------------------------------------------------------===//

// ImmLeaf
def immZExt2 : ImmLeaf<i32, [{return isUInt<2>(Imm);}]>;
def immZExt3 : ImmLeaf<i32, [{return isUInt<3>(Imm);}]>;
def immZExt4 : ImmLeaf<i32, [{return isUInt<4>(Imm);}]>;
def immZExt8 : ImmLeaf<i32, [{return isUInt<8>(Imm);}]>;
def immZExt10 : ImmLeaf<i32, [{return isUInt<10>(Imm);}]>;
def immSExt6 : ImmLeaf<i32, [{return isInt<6>(Imm);}]>;

// Patterns.
class DSPPat<dag pattern, dag result, Predicate pred = HasDSP> :
  Pat<pattern, result>, Requires<[pred]>;

class BitconvertPat<ValueType DstVT, ValueType SrcVT, RegisterClass DstRC,
                    RegisterClass SrcRC> :
   DSPPat<(DstVT (bitconvert (SrcVT SrcRC:$src))),
          (COPY_TO_REGCLASS SrcRC:$src, DstRC)>;

def : BitconvertPat<i32, v2i16, CPURegs, DSPRegs>;
def : BitconvertPat<i32, v4i8, CPURegs, DSPRegs>;
def : BitconvertPat<v2i16, i32, DSPRegs, CPURegs>;
def : BitconvertPat<v4i8, i32, DSPRegs, CPURegs>;

def : DSPPat<(v2i16 (load addr:$a)),
             (v2i16 (COPY_TO_REGCLASS (LW addr:$a), DSPRegs))>;
def : DSPPat<(v4i8 (load addr:$a)),
             (v4i8 (COPY_TO_REGCLASS (LW addr:$a), DSPRegs))>;
def : DSPPat<(store (v2i16 DSPRegs:$val), addr:$a),
             (SW (COPY_TO_REGCLASS DSPRegs:$val, CPURegs), addr:$a)>;
def : DSPPat<(store (v4i8 DSPRegs:$val), addr:$a),
             (SW (COPY_TO_REGCLASS DSPRegs:$val, CPURegs), addr:$a)>;
