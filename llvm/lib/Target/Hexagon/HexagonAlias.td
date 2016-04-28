//==- HexagonAlias.td - Hexagon Instruction Aliases ---------*- tablegen -*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
//                     Hexagon Instruction Mappings
//===----------------------------------------------------------------------===//

// V6_vassignp: Vector assign mapping.
let hasNewValue = 1, opNewValue = 0, isAsmParserOnly = 1 in
def HEXAGON_V6_vassignpair: CVI_VA_DV_Resource <
  (outs VecDblRegs:$Vdd),
  (ins VecDblRegs:$Vss),
  "$Vdd = $Vss">;

// maps Vd = #0 to Vd = vxor(Vd, Vd)
def : InstAlias<"$Vd = #0",
      (V6_vxor VectorRegs:$Vd, VectorRegs:$Vd, VectorRegs:$Vd)>,
      Requires<[HasV60T]>;

// maps Vdd  = #0 to Vdd = vsub(Vdd, Vdd)
def : InstAlias<"$Vdd = #0",
      (V6_vsubw_dv VecDblRegs:$Vdd, VecDblRegs:$Vdd, VecDblRegs:$Vdd)>,
      Requires<[HasV60T]>;
