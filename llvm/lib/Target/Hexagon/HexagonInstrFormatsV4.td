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
//                         Hexagon Intruction Flags +
//
//                        *** Must match BaseInfo.h ***
//----------------------------------------------------------------------------//

def TypeMEMOP  : IType<9>;
def TypeNV     : IType<10>;
def TypePREFIX : IType<30>;

//----------------------------------------------------------------------------//
//                         Intruction Classes Definitions +
//----------------------------------------------------------------------------//

//
// NV type instructions.
//
class NVInst<dag outs, dag ins, string asmstr, list<dag> pattern = [],
             string cstr = "">
  : InstHexagon<outs, ins, asmstr, pattern, cstr, NV_V4, TypeNV>;

class NVInst_V4<dag outs, dag ins, string asmstr, list<dag> pattern = [],
                string cstr = "">
  : NVInst<outs, ins, asmstr, pattern, cstr>;

// Definition of Post increment new value store.
class NVInstPost_V4<dag outs, dag ins, string asmstr, list<dag> pattern = [],
               string cstr = "">
  : NVInst<outs, ins, asmstr, pattern, cstr>;

// Post increment ST Instruction.
let mayStore = 1 in
class NVInstPI_V4<dag outs, dag ins, string asmstr, list<dag> pattern = [],
               string cstr = "">
  : NVInst<outs, ins, asmstr, pattern, cstr>;

// New-value conditional branch.
class NCJInst<dag outs, dag ins, string asmstr, list<dag> pattern = [],
              string cstr = "">
  : NVInst<outs, ins, asmstr, pattern, cstr>;

let mayLoad = 1, mayStore = 1 in
class MEMInst<dag outs, dag ins, string asmstr, list<dag> pattern = [],
              string cstr = "">
  : InstHexagon<outs, ins, asmstr, pattern, cstr, MEM_V4, TypeMEMOP>;

class MEMInst_V4<dag outs, dag ins, string asmstr, list<dag> pattern = [],
                 string cstr = "">
  : MEMInst<outs, ins, asmstr, pattern, cstr>;

let isCodeGenOnly = 1 in
class EXTENDERInst<dag outs, dag ins, string asmstr, list<dag> pattern = []>
  : InstHexagon<outs, ins, asmstr, pattern, "", PREFIX, TypePREFIX>;
