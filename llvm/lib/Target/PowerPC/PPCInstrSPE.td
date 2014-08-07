//=======-- PPCInstrSPE.td - The PowerPC SPE Extension -*- tablegen -*-=======//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file describes the Signal Processing Engine extension to
// the PowerPC instruction set.
//
//===----------------------------------------------------------------------===//

class EVXForm_1<bits<11> xo, dag OOL, dag IOL, string asmstr,
               InstrItinClass itin> : I<4, OOL, IOL, asmstr, itin> {
  bits<5> RT;
  bits<5> RA;
  bits<5> RB;

  let Pattern = [];
  
  let Inst{6-10}  = RT;
  let Inst{11-15} = RA;
  let Inst{16-20} = RB;
  let Inst{21-31} = xo;
}

let Predicates = [HasSPE], isAsmParserOnly = 1 in {

def EVMRA : EVXForm_1<1220, (outs gprc:$RT), (ins gprc:$RA),
                      "evmra $RT, $RA", IIC_VecFP> {
  let RB = 0;
}

def EVLDDX         : EVXForm_1<768, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "evlddx $RT, $RA, $RB", IIC_VecFP>;
def EVLDWX         : EVXForm_1<770, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "evldwx $RT, $RA, $RB", IIC_VecFP>;
def EVLDHX         : EVXForm_1<772, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "evldhx $RT, $RA, $RB", IIC_VecFP>;
def EVLHHESPLATX   : EVXForm_1<776, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "evlhhesplatx $RT, $RA, $RB", IIC_VecFP>;
def EVLHHOUSPLATX  : EVXForm_1<780, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "evlhhousplatx $RT, $RA, $RB", IIC_VecFP>;
def EVLHHOSSPLATX  : EVXForm_1<782, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "evlhhossplatx $RT, $RA, $RB", IIC_VecFP>;
def EVLWHEX        : EVXForm_1<784, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "evlwhex $RT, $RA, $RB", IIC_VecFP>;
def EVLWHOUX       : EVXForm_1<788, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "evlwhoux $RT, $RA, $RB", IIC_VecFP>;
def EVLWHOSX       : EVXForm_1<790, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "evlwhosx $RT, $RA, $RB", IIC_VecFP>;
def EVLWWSPLATX    : EVXForm_1<792, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "evlwwsplatx $RT, $RA, $RB", IIC_VecFP>;
def EVLWHSPLATX    : EVXForm_1<796, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "evlwhsplatx $RT, $RA, $RB", IIC_VecFP>;

def EVMERGEHI      : EVXForm_1<556, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "evmergehi $RT, $RA, $RB", IIC_VecFP>;
def EVMERGELO      : EVXForm_1<557, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "evmergelo $RT, $RA, $RB", IIC_VecFP>;
def EVMERGEHILO    : EVXForm_1<558, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "evmergehilo $RT, $RA, $RB", IIC_VecFP>;
def EVMERGELOHI    : EVXForm_1<559, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "evmergelohi $RT, $RA, $RB", IIC_VecFP>;

} // HasSPE
