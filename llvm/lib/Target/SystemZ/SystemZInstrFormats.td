//===- SystemZInstrFormats.td - SystemZ Instruction Formats ----*- tablegen -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//

class InstSystemZ<dag outs, dag ins, string asmstr, list<dag> pattern> : Instruction {
  let Namespace = "SystemZ";

  dag OutOperandList = outs;
  dag InOperandList = ins;
  let AsmString   = asmstr;
  let Pattern = pattern;
}

//===----------------------------------------------------------------------===//
// E format
//===----------------------------------------------------------------------===//

class F_E<bits<16> opcode,
          dag outs, dag ins, string asmstr, list<dag> pattern>
   : InstSystemZ<outs, ins, asmstr, pattern> {

  field bits<16> Inst;

  let Inst{15-0} = opcode;   
}

//===----------------------------------------------------------------------===//
// I format
//===----------------------------------------------------------------------===//

class F_I<bits<16> opcode,
          dag outs, dag ins, string asmstr, list<dag> pattern>
   : InstSystemZ<outs, ins, asmstr, pattern> {

  field bits<48> Inst;

  let Inst{47-32} = opcode;   
  //let Inst{31-0}  = simm32;
}

//===----------------------------------------------------------------------===//
// RR format
//===----------------------------------------------------------------------===//

class F_RR<bits<8> opcode,
           dag outs, dag ins, string asmstr, list<dag> pattern>
   : InstSystemZ<outs, ins, asmstr, pattern> {

  field bits<16> Inst;

  let Inst{15-8} = opcode;
}

//===----------------------------------------------------------------------===//
// RRE format
//===----------------------------------------------------------------------===//

class F_RRE<bits<16> opcode,
            dag outs, dag ins, string asmstr, list<dag> pattern>
   : InstSystemZ<outs, ins, asmstr, pattern> {

  field bits<32> Inst;

  let Inst{31-16} = opcode;
  let Inst{15-8}  = 0;
  //let Inst{7-4}   = r1;
  //let Inst{3-0}   = r2;
}

//===----------------------------------------------------------------------===//
// RRF format (1)
//===----------------------------------------------------------------------===//

class F_RRF_1<bits<16> opcode,
              dag outs, dag ins, string asmstr, list<dag> pattern>
   : InstSystemZ<outs, ins, asmstr, pattern> {

  field bits<32> Inst;

  let Inst{31-16} = opcode;
  //let Inst{15-12} = r1;
  let Inst{11-8}  = 0;
  //let Inst{7-4}   = r3;
  //let Inst{3-0}   = r2;
}

//===----------------------------------------------------------------------===//
// RRF format (2)
//===----------------------------------------------------------------------===//

class F_RRF_2<bits<16> opcode,
              dag outs, dag ins, string asmstr, list<dag> pattern>
   : InstSystemZ<outs, ins, asmstr, pattern> {

  field bits<32> Inst;

  let Inst{31-16} = opcode;
  //let Inst{15-12} = m3;
  let Inst{11-8}  = 0;
  //let Inst{7-4}   = r1;
  //let Inst{3-0}   = r2;
}

//===----------------------------------------------------------------------===//
// RRF format (3)
//===----------------------------------------------------------------------===//

class F_RRF_3<bits<16> opcode,
              dag outs, dag ins, string asmstr, list<dag> pattern>
   : InstSystemZ<outs, ins, asmstr, pattern> {

  field bits<32> Inst;

  let Inst{31-16} = opcode;
  //let Inst{15-12} = r3;
  //let Inst{11-8}  = m4;
  //let Inst{7-4}   = r1;
  //let Inst{3-0}   = r2;
}

//===----------------------------------------------------------------------===//
// RX format
//===----------------------------------------------------------------------===//

class F_RX<bits<8> opcode,
           dag outs, dag ins, string asmstr, list<dag> pattern>
   : InstSystemZ<outs, ins, asmstr, pattern> {

  field bits<32> Inst;

  let Inst{31-24} = opcode;
  //let Inst{23-20} = r1;
  //let Inst{19-16} = x2;
  //let Inst{15-12} = b2;
  //let Inst{11-0}  = udisp12;
}

//===----------------------------------------------------------------------===//
// RXE format
//===----------------------------------------------------------------------===//

class F_RXE<bits<8> opcode,
            dag outs, dag ins, string asmstr, list<dag> pattern>
   : InstSystemZ<outs, ins, asmstr, pattern> {

  field bits<48> Inst;

  let Inst{47-40} = opcode;
  //let Inst{39-36} = r1;
  //let Inst{35-32} = x2;
  //let Inst{31-28} = b2;
  //let Inst{27-16} = udisp12;
  let Inst{15-8}  = 0;
  //let Inst{7-0}   = op2;
}

//===----------------------------------------------------------------------===//
// RXF format
//===----------------------------------------------------------------------===//

class F_RXF<bits<8> opcode,
            dag outs, dag ins, string asmstr, list<dag> pattern>
   : InstSystemZ<outs, ins, asmstr, pattern> {

  field bits<48> Inst;

  let Inst{47-40} = opcode;
  //let Inst{39-36} = r3;
  //let Inst{35-32} = x2;
  //let Inst{31-28} = b2;
  //let Inst{27-16} = udisp12;
  //let Inst{15-11} = r1;
  let Inst{11-8}  = 0;
  //let Inst{7-0}   = op2;
}

//===----------------------------------------------------------------------===//
// RXY format
//===----------------------------------------------------------------------===//

class F_RXY<bits<8> opcode,
            dag outs, dag ins, string asmstr, list<dag> pattern>
   : InstSystemZ<outs, ins, asmstr, pattern> {

  field bits<48> Inst;

  let Inst{47-40} = opcode;
  //let Inst{39-36} = r1;
  //let Inst{35-32} = x2;
  //let Inst{31-28} = b2;
  //let Inst{27-8}  = sdisp20;
  //let Inst{7-0}   = op2;
}

//===----------------------------------------------------------------------===//
// RS format (1)
//===----------------------------------------------------------------------===//

class F_RS_1<bits<8> opcode,
             dag outs, dag ins, string asmstr, list<dag> pattern>
   : InstSystemZ<outs, ins, asmstr, pattern> {

  field bits<32> Inst;

  let Inst{31-24} = opcode;
  //let Inst{23-20} = r1;
  //let Inst{19-16} = r3;
  //let Inst{15-12} = b2;
  //let Inst{11-0}  = udisp12;
}

//===----------------------------------------------------------------------===//
// RS format (2)
//===----------------------------------------------------------------------===//

class F_RS_2<bits<8> opcode,
             dag outs, dag ins, string asmstr, list<dag> pattern>
   : InstSystemZ<outs, ins, asmstr, pattern> {

  field bits<32> Inst;

  let Inst{31-24} = opcode;
  //let Inst{23-20} = r1;
  //let Inst{19-16} = m3;
  //let Inst{15-12} = b2;
  //let Inst{11-0}  = udisp12;
}

//===----------------------------------------------------------------------===//
// RS format (3)
//===----------------------------------------------------------------------===//

class F_RS_3<bits<8> opcode,
             dag outs, dag ins, string asmstr, list<dag> pattern>
   : InstSystemZ<outs, ins, asmstr, pattern> {

  field bits<32> Inst;

  let Inst{31-24} = opcode;
  //let Inst{23-20} = r1;
  let Inst{19-16} = 0;
  //let Inst{15-12} = b2;
  //let Inst{11-0}  = udisp12;
}

//===----------------------------------------------------------------------===//
// RSY format (1)
//===----------------------------------------------------------------------===//

class F_RSY_1<bits<8> opcode,
              dag outs, dag ins, string asmstr, list<dag> pattern>
   : InstSystemZ<outs, ins, asmstr, pattern> {

  field bits<48> Inst;

  let Inst{47-40} = opcode;
  //let Inst{39-36} = r1;
  //let Inst{35-32} = r3;
  //let Inst{31-28} = b2;
  //let Inst{27-8}  = sdisp20;
  //let Inst{7-0}   = op2;
}

//===----------------------------------------------------------------------===//
// RSY format (2)
//===----------------------------------------------------------------------===//

class F_RSY_2<bits<8> opcode,
              dag outs, dag ins, string asmstr, list<dag> pattern>
   : InstSystemZ<outs, ins, asmstr, pattern> {

  field bits<48> Inst;

  let Inst{47-40} = opcode;
  //let Inst{39-36} = r1;
  //let Inst{35-32} = m3;
  //let Inst{31-28} = b2;
  //let Inst{27-8}  = sdisp20;
  //let Inst{7-0}   = op2;
}

//===----------------------------------------------------------------------===//
// RSL format
//===----------------------------------------------------------------------===//

class F_RSL<bits<8> opcode,
            dag outs, dag ins, string asmstr, list<dag> pattern>
   : InstSystemZ<outs, ins, asmstr, pattern> {

  field bits<48> Inst;

  let Inst{47-40} = opcode;
  //let Inst{39-36} = ll;
  let Inst{35-32} = 0;
  //let Inst{31-28} = b1;
  //let Inst{27-16} = udisp12;
  let Inst{15-8}  = 0;
  //let Inst{7-0}   = op2;
}

//===----------------------------------------------------------------------===//
// RSI format
//===----------------------------------------------------------------------===//

class F_RSI<bits<8> opcode,
            dag outs, dag ins, string asmstr, list<dag> pattern>
   : InstSystemZ<outs, ins, asmstr, pattern> {

  field bits<32> Inst;

  let Inst{31-24} = opcode;
  //let Inst{23-20} = r1;
  //let Inst{19-16} = r3;
  //let Inst{15-0}  = simm16;
}

//===----------------------------------------------------------------------===//
// RI format
//===----------------------------------------------------------------------===//

class F_RI<bits<8> opcode,
           dag outs, dag ins, string asmstr, list<dag> pattern>
   : InstSystemZ<outs, ins, asmstr, pattern> {

  field bits<32> Inst;

  let Inst{31-24} = opcode;
  //let Inst{23-20} = r1;
  //let Inst{19-16} = op2;
  //let Inst{15-0}  = simm16;
}

//===----------------------------------------------------------------------===//
// RIE format
//===----------------------------------------------------------------------===//

class F_RIE<bits<8> opcode,
            dag outs, dag ins, string asmstr, list<dag> pattern>
   : InstSystemZ<outs, ins, asmstr, pattern> {

  field bits<48> Inst;

  let Inst{47-40} = opcode;
  //let Inst{39-36} = r1;
  //let Inst{35-32} = r2;
  //let Inst{31-16} = simm16;
  let Inst{15-8}  = 0;
  //let Inst{7-0}   = op2;
}

//===----------------------------------------------------------------------===//
// RIL format (1)
//===----------------------------------------------------------------------===//

class F_RIL_1<bits<8> opcode,
              dag outs, dag ins, string asmstr, list<dag> pattern>
   : InstSystemZ<outs, ins, asmstr, pattern> {

  field bits<48> Inst;

  let Inst{47-40} = opcode;
  //let Inst{39-36} = r1;
  //let Inst{35-32} = op2;
  //let Inst{31-0}  = simm32;
}

//===----------------------------------------------------------------------===//
// RIL format (2)
//===----------------------------------------------------------------------===//

class F_RIL_2<bits<8> opcode,
              dag outs, dag ins, string asmstr, list<dag> pattern>
   : InstSystemZ<outs, ins, asmstr, pattern> {

  field bits<48> Inst;

  let Inst{47-40} = opcode;
  //let Inst{39-36} = m1;
  //let Inst{35-32} = op2;
  //let Inst{31-0}  = simm32;
}

//===----------------------------------------------------------------------===//
// SI format
//===----------------------------------------------------------------------===//

class F_SI<bits<8> opcode,
           dag outs, dag ins, string asmstr, list<dag> pattern>
   : InstSystemZ<outs, ins, asmstr, pattern> {

  field bits<32> Inst;

  let Inst{31-24} = opcode;
  //let Inst{23-16} = simm8;
  //let Inst{15-12} = b1;
  //let Inst{11-0}  = udisp12;
}

//===----------------------------------------------------------------------===//
// SIY format
//===----------------------------------------------------------------------===//

class F_SIY<bits<8> opcode,
            dag outs, dag ins, string asmstr, list<dag> pattern>
   : InstSystemZ<outs, ins, asmstr, pattern> {

  field bits<48> Inst;

  let Inst{47-40} = opcode;
  //let Inst{39-32} = simm8;
  //let Inst{31-28} = b1;
  //let Inst{27-8}  = sdisp20;
  //let Inst{7-0}   = op2;
}

//===----------------------------------------------------------------------===//
// S format
//===----------------------------------------------------------------------===//

class F_S<bits<16> opcode,
          dag outs, dag ins, string asmstr, list<dag> pattern>
   : InstSystemZ<outs, ins, asmstr, pattern> {

  field bits<32> Inst;

  let Inst{31-16} = opcode;
  //let Inst{15-12} = b2;
  //let Inst{11-0}  = udisp12;
}

//===----------------------------------------------------------------------===//
// SS format (1)
//===----------------------------------------------------------------------===//

class F_SS_1<bits<8> opcode,
             dag outs, dag ins, string asmstr, list<dag> pattern>
   : InstSystemZ<outs, ins, asmstr, pattern> {

  field bits<48> Inst;

  let Inst{47-40} = opcode;
  //let Inst{39-32} = ll;
  //let Inst{31-28} = b1;
  //let Inst{27-16} = udisp12;
  //let Inst{15-12} = b2;
  //let Inst{11-0}  = udisp12_2;
}

//===----------------------------------------------------------------------===//
// SS format (2)
//===----------------------------------------------------------------------===//

class F_SS_2<bits<8> opcode,
             dag outs, dag ins, string asmstr, list<dag> pattern>
   : InstSystemZ<outs, ins, asmstr, pattern> {

  field bits<48> Inst;

  let Inst{47-40} = opcode;
  //let Inst{39-36} = l1;
  //let Inst{35-32} = l2;
  //let Inst{31-28} = b1;
  //let Inst{27-16} = udisp12;
  //let Inst{15-12} = b2;
  //let Inst{11-0}  = udisp12_2;
}

//===----------------------------------------------------------------------===//
// SS format (3)
//===----------------------------------------------------------------------===//

class F_SS_3<bits<8> opcode,
             dag outs, dag ins, string asmstr, list<dag> pattern>
   : InstSystemZ<outs, ins, asmstr, pattern> {

  field bits<48> Inst;

  let Inst{47-40} = opcode;
  //let Inst{39-36} = r1;
  //let Inst{35-32} = r3;
  //let Inst{31-28} = b1;
  //let Inst{27-16} = udisp12;
  //let Inst{15-12} = b2;
  //let Inst{11-0}  = udisp12_2;
}

//===----------------------------------------------------------------------===//
// SS format (4)
//===----------------------------------------------------------------------===//

class F_SS_4<bits<8> opcode,
             dag outs, dag ins, string asmstr, list<dag> pattern>
   : InstSystemZ<outs, ins, asmstr, pattern> {

  field bits<48> Inst;

  let Inst{47-40} = opcode;
  //let Inst{39-36} = r1;
  //let Inst{35-32} = r3;
  //let Inst{31-28} = b2;
  //let Inst{27-16} = udisp12_2;
  //let Inst{15-12} = b4;
  //let Inst{11-0}  = udisp12_4;
}

//===----------------------------------------------------------------------===//
// SSE format
//===----------------------------------------------------------------------===//

class F_SSE<bits<16> opcode,
            dag outs, dag ins, string asmstr, list<dag> pattern>
   : InstSystemZ<outs, ins, asmstr, pattern> {

  field bits<48> Inst;

  let Inst{47-32} = opcode;
  //let Inst{31-28} = b1;
  //let Inst{27-16} = udisp12;
  //let Inst{15-12} = b2;
  //let Inst{11-0}  = udisp12_2;
}

//===----------------------------------------------------------------------===//
// Pseudo instructions
//===----------------------------------------------------------------------===//

class Pseudo<dag outs, dag ins, string asmstr, list<dag> pattern>
   : InstSystemZ<outs, ins, asmstr, pattern> {
}
