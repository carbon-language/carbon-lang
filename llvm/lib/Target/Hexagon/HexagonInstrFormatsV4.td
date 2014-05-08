//==- HexagonInstrFormats.td - Hexagon Instruction Formats --*- tablegen -*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file describes the Hexagon V4 instruction classes in TableGen format.
//
//===----------------------------------------------------------------------===//

//----------------------------------------------------------------------------//
//                         Hexagon Instruction Flags
//
//                        *** Must match BaseInfo.h ***
//----------------------------------------------------------------------------//

def TypeMEMOP  : IType<9>;
def TypeNV     : IType<10>;
def TypePREFIX : IType<30>;

//----------------------------------------------------------------------------//
//                         Instruction Classes Definitions
//----------------------------------------------------------------------------//

//
// NV type instructions.
//
class NVInst<dag outs, dag ins, string asmstr, list<dag> pattern = [],
             string cstr = "", InstrItinClass itin = NCJ_tc_3or4stall_SLOT0>
  : InstHexagon<outs, ins, asmstr, pattern, cstr, itin, TypeNV>;

class NVInst_V4<dag outs, dag ins, string asmstr, list<dag> pattern = [],
                string cstr = "", InstrItinClass itin = NCJ_tc_3or4stall_SLOT0>
  : NVInst<outs, ins, asmstr, pattern, cstr, itin>;

// Definition of Post increment new value store.
class NVInstPost_V4<dag outs, dag ins, string asmstr, list<dag> pattern = [],
               string cstr = "", InstrItinClass itin = ST_tc_st_SLOT0>
  : NVInst<outs, ins, asmstr, pattern, cstr, itin>;

// Post increment ST Instruction.
let mayStore = 1 in
class NVInstPI_V4<dag outs, dag ins, string asmstr, list<dag> pattern = [],
               string cstr = "", InstrItinClass itin = ST_tc_st_SLOT0>
  : NVInst<outs, ins, asmstr, pattern, cstr, itin>;

// New-value conditional branch.
class NCJInst<dag outs, dag ins, string asmstr, list<dag> pattern = [],
              string cstr = "">
  : NVInst<outs, ins, asmstr, pattern, cstr>;

let mayLoad = 1, mayStore = 1 in
class MEMInst<dag outs, dag ins, string asmstr, list<dag> pattern = [],
              string cstr = "", InstrItinClass itin = V4LDST_tc_st_SLOT0>
  : InstHexagon<outs, ins, asmstr, pattern, cstr, itin, TypeMEMOP>;

class MEMInst_V4<dag outs, dag ins, string asmstr, list<dag> pattern = [],
                 string cstr = "", InstrItinClass itin = V4LDST_tc_st_SLOT0>
  : MEMInst<outs, ins, asmstr, pattern, cstr, itin>;

let isCodeGenOnly = 1 in
class EXTENDERInst<dag outs, dag ins, string asmstr, list<dag> pattern = []>
  : InstHexagon<outs, ins, asmstr, pattern, "", EXTENDER_tc_1_SLOT0123,
                TypePREFIX>;
