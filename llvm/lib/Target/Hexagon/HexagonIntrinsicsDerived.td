//===-- HexagonIntrinsicsDerived.td - Derived intrinsics ---*- tablegen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Multiply 64-bit and use lower result
//
// Optimized with intrinisics accumulates
//
def : Pat <(mul DoubleRegs:$src1, DoubleRegs:$src2),
      (COMBINE_rr
                  (Hexagon_M2_maci
                           (Hexagon_M2_maci (EXTRACT_SUBREG  (MPYU64 (EXTRACT_SUBREG DoubleRegs:$src1, subreg_loreg),
                                                           (EXTRACT_SUBREG DoubleRegs:$src2, subreg_loreg)),
                                       subreg_hireg),
                                       (EXTRACT_SUBREG DoubleRegs:$src1, subreg_loreg),
                                       (EXTRACT_SUBREG DoubleRegs:$src2, subreg_hireg)),
                            (EXTRACT_SUBREG DoubleRegs:$src2, subreg_loreg),
                            (EXTRACT_SUBREG DoubleRegs:$src1, subreg_hireg)),
                    (EXTRACT_SUBREG  (MPYU64 (EXTRACT_SUBREG DoubleRegs:$src1, subreg_loreg),
                                      (EXTRACT_SUBREG DoubleRegs:$src2, subreg_loreg)),
                     subreg_loreg))>;



