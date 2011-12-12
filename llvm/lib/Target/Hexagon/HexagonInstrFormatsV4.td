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

//
// NV type instructions.
//
class NVInst_V4<dag outs, dag ins, string asmstr, list<dag> pattern>
  : InstHexagon<outs, ins, asmstr, pattern, "", NV_V4> {
  bits<5> rd;
  bits<5> rs;
  bits<13> imm13;
}

// Definition of Post increment new value store.
class NVInstPost_V4<dag outs, dag ins, string asmstr, list<dag> pattern,
                    string cstr>
  : InstHexagon<outs, ins, asmstr, pattern, cstr, NV_V4> {
  bits<5> rd;
  bits<5> rs;
  bits<5> rt;
  bits<13> imm13;
}

// Post increment ST Instruction.
class NVInstPI_V4<dag outs, dag ins, string asmstr, list<dag> pattern,
                  string cstr>
  : NVInstPost_V4<outs, ins, asmstr, pattern, cstr> {
  let rt{0-4} = 0;
}

class MEMInst_V4<dag outs, dag ins, string asmstr, list<dag> pattern>
  : InstHexagon<outs, ins, asmstr, pattern, "",  MEM_V4> {
  bits<5> rd;
  bits<5> rs;
  bits<6> imm6;
}
