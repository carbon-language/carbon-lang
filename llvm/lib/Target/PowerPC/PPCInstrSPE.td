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

class EFXForm_1<bits<11> xo, dag OOL, dag IOL, string asmstr,
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

class EFXForm_2<bits<11> xo, dag OOL, dag IOL, string asmstr,
               InstrItinClass itin> : EFXForm_1<xo, OOL, IOL, asmstr, itin> {
  let RB = 0;
}

class EFXForm_2a<bits<11> xo, dag OOL, dag IOL, string asmstr,
               InstrItinClass itin> : EFXForm_1<xo, OOL, IOL, asmstr, itin> {
  let RA = 0;
}

class EFXForm_3<bits<11> xo, dag OOL, dag IOL, string asmstr,
              InstrItinClass itin> : I<4, OOL, IOL, asmstr, itin> {
  bits<3> crD;
  bits<5> RA;
  bits<5> RB;

  let Pattern = [];

  let Inst{6-8}  = crD;
  let Inst{9-10}  = 0;
  let Inst{11-15} = RA;
  let Inst{16-20} = RB;
  let Inst{21-31} = xo;
}

class EVXForm_1<bits<11> xo, dag OOL, dag IOL, string asmstr,
               InstrItinClass itin> : I<4, OOL, IOL, asmstr, itin> {
  bits<5> RT;
  bits<5> RA;
  bits<5> RB;

  let Inst{6-10}  = RT;
  let Inst{11-15} = RA;
  let Inst{16-20} = RB;
  let Inst{21-31} = xo;
}

class EVXForm_2<bits<11> xo, dag OOL, dag IOL, string asmstr,
               InstrItinClass itin> :
               EVXForm_1<xo, OOL, IOL, asmstr, itin> {
  let RB = 0;
}

class EVXForm_2a<bits<11> xo, dag OOL, dag IOL, string asmstr,
               InstrItinClass itin> :
               EVXForm_1<xo, OOL, IOL, asmstr, itin> {
  let RA = 0;
}

class EVXForm_3<bits<11> xo, dag OOL, dag IOL, string asmstr,
              InstrItinClass itin> :
              I<4, OOL, IOL, asmstr, itin> {
  bits<3> crD;
  bits<5> RA;
  bits<5> RB;

  let Inst{6-8}  = crD;
  let Inst{9-10}  = 0;
  let Inst{11-15} = RA;
  let Inst{16-20} = RB;
  let Inst{21-31} = xo;
}

class EVXForm_4<bits<8> xo, dag OOL, dag IOL, string asmstr,
               InstrItinClass itin> :
               I<4, OOL, IOL, asmstr, itin> {
  bits<3> crD;
  bits<5> RA;
  bits<5> RB;
  bits<5> RT;

  let Inst{6-10}  = RT;
  let Inst{11-15} = RA;
  let Inst{16-20} = RB;
  let Inst{21-28} = xo;
  let Inst{29-31} = crD;
}

class EVXForm_D<bits<11> xo, dag OOL, dag IOL, string asmstr,
               InstrItinClass itin> : I<4, OOL, IOL, asmstr, itin> {
  bits<5> RT;
  bits<21> D;

  let Pattern = [];

  let Inst{6-10}  = RT;
  let Inst{20} = D{0};
  let Inst{19} = D{1};
  let Inst{18} = D{2};
  let Inst{17} = D{3};
  let Inst{16} = D{4};
  let Inst{15} = D{5};
  let Inst{14} = D{6};
  let Inst{13} = D{7};
  let Inst{12} = D{8};
  let Inst{11} = D{9};
  let Inst{11-20} = D{0-9};
  let Inst{21-31} = xo;
}

let DecoderNamespace = "SPE", Predicates = [HasSPE] in {

def BRINC          : EVXForm_1<527, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "brinc $RT, $RA, $RB", IIC_IntSimple>;

// Double-precision floating point
def EFDABS         : EFXForm_2<740, (outs gprc:$RT), (ins gprc:$RA),
                                "efdabs $RT, $RA", IIC_FPDGeneral>;

def EFDADD         : EFXForm_1<736, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                                "efdadd $RT, $RA, $RB", IIC_FPAddSub>;

def EFDCFS         : EFXForm_2a<751, (outs gprc:$RT), (ins gprc:$RB),
                                "efdcfs $RT, $RB", IIC_FPDGeneral>;

def EFDCFSF        : EFXForm_2a<755, (outs gprc:$RT), (ins gprc:$RB),
                                "efdcfsf $RT, $RB", IIC_FPDGeneral>;

def EFDCFSI        : EFXForm_2a<753, (outs gprc:$RT), (ins gprc:$RB),
                                "efdcfsi $RT, $RB", IIC_FPDGeneral>;

def EFDCFSID       : EFXForm_2a<739, (outs gprc:$RT), (ins gprc:$RB),
                                "efdcfsid $RT, $RB", IIC_FPDGeneral>;

def EFDCFUF        : EFXForm_2a<754, (outs gprc:$RT), (ins gprc:$RB),
                                "efdcfuf $RT, $RB", IIC_FPDGeneral>;

def EFDCFUI        : EFXForm_2a<752, (outs gprc:$RT), (ins gprc:$RB),
                                "efdcfui $RT, $RB", IIC_FPDGeneral>;

def EFDCFUID       : EFXForm_2a<738, (outs gprc:$RT), (ins gprc:$RB),
                                "efdcfuid $RT, $RB", IIC_FPDGeneral>;

let isCompare = 1 in {
def EFDCMPEQ       : EFXForm_3<750, (outs crrc:$crD), (ins gprc:$RA, gprc:$RB),
                               "efdcmpeq $crD, $RA, $RB", IIC_FPDGeneral>;
def EFDCMPGT       : EFXForm_3<748, (outs crrc:$crD), (ins gprc:$RA, gprc:$RB),
                               "efdcmpgt $crD, $RA, $RB", IIC_FPDGeneral>;
def EFDCMPLT       : EFXForm_3<749, (outs crrc:$crD), (ins gprc:$RA, gprc:$RB),
                               "efdcmplt $crD, $RA, $RB", IIC_FPDGeneral>;
}

def EFDCTSF        : EFXForm_2a<759, (outs gprc:$RT), (ins gprc:$RB),
                                "efdctsf $RT, $RB", IIC_FPDGeneral>;

def EFDCTSI        : EFXForm_2a<757, (outs gprc:$RT), (ins gprc:$RB),
                                "efdctsi $RT, $RB", IIC_FPDGeneral>;

def EFDCTSIDZ      : EFXForm_2a<747, (outs gprc:$RT), (ins gprc:$RB),
                                "efdctsidz $RT, $RB", IIC_FPDGeneral>;

def EFDCTSIZ       : EFXForm_2a<762, (outs gprc:$RT), (ins gprc:$RB),
                                "efdctsiz $RT, $RB", IIC_FPDGeneral>;

def EFDCTUF        : EFXForm_2a<758, (outs gprc:$RT), (ins gprc:$RB),
                                "efdctuf $RT, $RB", IIC_FPDGeneral>;

def EFDCTUI        : EFXForm_2a<756, (outs gprc:$RT), (ins gprc:$RB),
                                "efdctui $RT, $RB", IIC_FPDGeneral>;

def EFDCTUIDZ      : EFXForm_2a<746, (outs gprc:$RT), (ins gprc:$RB),
                                "efdctuidz $RT, $RB", IIC_FPDGeneral>;

def EFDCTUIZ       : EFXForm_2a<760, (outs gprc:$RT), (ins gprc:$RB),
                                "efdctuiz $RT, $RB", IIC_FPDGeneral>;

def EFDDIV         : EFXForm_1<745, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "efddiv $RT, $RA, $RB", IIC_FPDivD>;

def EFDMUL         : EFXForm_1<744, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "efdmul $RT, $RA, $RB", IIC_FPDGeneral>;

def EFDNABS        : EFXForm_2<741, (outs gprc:$RT), (ins gprc:$RA),
                               "efdnabs $RT, $RA", IIC_FPDGeneral>;

def EFDNEG         : EFXForm_2<742, (outs gprc:$RT), (ins gprc:$RA),
                               "efdneg $RT, $RA", IIC_FPDGeneral>;

def EFDSUB         : EFXForm_1<737, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "efdsub $RT, $RA, $RB", IIC_FPDGeneral>;

let isCompare = 1 in {
def EFDTSTEQ       : EFXForm_3<766, (outs crrc:$crD), (ins gprc:$RA, gprc:$RB),
                               "efdtsteq $crD, $RA, $RB", IIC_FPDGeneral>;
def EFDTSTGT       : EFXForm_3<764, (outs crrc:$crD), (ins gprc:$RA, gprc:$RB),
                               "efdtstgt $crD, $RA, $RB", IIC_FPDGeneral>;
def EFDTSTLT       : EFXForm_3<765, (outs crrc:$crD), (ins gprc:$RA, gprc:$RB),
                               "efdtstlt $crD, $RA, $RB", IIC_FPDGeneral>;
}

// Single-precision floating point
def EFSABS         : EFXForm_2<708, (outs gprc:$RT), (ins gprc:$RA),
                                "efsabs $RT, $RA", IIC_FPSGeneral>;

def EFSADD         : EFXForm_1<704, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                                "efsadd $RT, $RA, $RB", IIC_FPAddSub>;

def EFSCFD         : EFXForm_2a<719, (outs gprc:$RT), (ins gprc:$RB),
                                "efscfd $RT, $RB", IIC_FPSGeneral>;

def EFSCFSF        : EFXForm_2a<723, (outs gprc:$RT), (ins gprc:$RB),
                                "efscfsf $RT, $RB", IIC_FPSGeneral>;

def EFSCFSI        : EFXForm_2a<721, (outs gprc:$RT), (ins gprc:$RB),
                                "efscfsi $RT, $RB", IIC_FPSGeneral>;

def EFSCFUF        : EFXForm_2a<722, (outs gprc:$RT), (ins gprc:$RB),
                                "efscfuf $RT, $RB", IIC_FPSGeneral>;

def EFSCFUI        : EFXForm_2a<720, (outs gprc:$RT), (ins gprc:$RB),
                                "efscfui $RT, $RB", IIC_FPSGeneral>;

let isCompare = 1 in {
def EFSCMPEQ       : EFXForm_3<718, (outs crrc:$crD), (ins gprc:$RA, gprc:$RB),
                               "efscmpeq $crD, $RA, $RB", IIC_FPCompare>;
def EFSCMPGT       : EFXForm_3<716, (outs crrc:$crD), (ins gprc:$RA, gprc:$RB),
                               "efscmpgt $crD, $RA, $RB", IIC_FPCompare>;
def EFSCMPLT       : EFXForm_3<717, (outs crrc:$crD), (ins gprc:$RA, gprc:$RB),
                               "efscmplt $crD, $RA, $RB", IIC_FPCompare>;
}

def EFSCTSF        : EFXForm_2a<727, (outs gprc:$RT), (ins gprc:$RB),
                                "efsctsf $RT, $RB", IIC_FPSGeneral>;

def EFSCTSI        : EFXForm_2a<725, (outs gprc:$RT), (ins gprc:$RB),
                                "efsctsi $RT, $RB", IIC_FPSGeneral>;

def EFSCTSIZ       : EFXForm_2a<730, (outs gprc:$RT), (ins gprc:$RB),
                                "efsctsiz $RT, $RB", IIC_FPSGeneral>;

def EFSCTUF        : EFXForm_2a<726, (outs gprc:$RT), (ins gprc:$RB),
                                "efsctuf $RT, $RB", IIC_FPSGeneral>;

def EFSCTUI        : EFXForm_2a<724, (outs gprc:$RT), (ins gprc:$RB),
                                "efsctui $RT, $RB", IIC_FPSGeneral>;

def EFSCTUIZ       : EFXForm_2a<728, (outs gprc:$RT), (ins gprc:$RB),
                                "efsctuiz $RT, $RB", IIC_FPSGeneral>;

def EFSDIV         : EFXForm_1<713, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "efsdiv $RT, $RA, $RB", IIC_FPDivD>;

def EFSMUL         : EFXForm_1<712, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "efsmul $RT, $RA, $RB", IIC_FPGeneral>;

def EFSNABS        : EFXForm_2<709, (outs gprc:$RT), (ins gprc:$RA),
                               "efsnabs $RT, $RA", IIC_FPGeneral>;

def EFSNEG         : EFXForm_2<710, (outs gprc:$RT), (ins gprc:$RA),
                               "efsneg $RT, $RA", IIC_FPGeneral>;

def EFSSUB         : EFXForm_1<705, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "efssub $RT, $RA, $RB", IIC_FPSGeneral>;

let isCompare = 1 in {
def EFSTSTEQ       : EFXForm_3<734, (outs crrc:$crD), (ins gprc:$RA, gprc:$RB),
                               "efststeq $crD, $RA, $RB", IIC_FPCompare>;
def EFSTSTGT       : EFXForm_3<732, (outs crrc:$crD), (ins gprc:$RA, gprc:$RB),
                               "efststgt $crD, $RA, $RB", IIC_FPCompare>;
def EFSTSTLT       : EFXForm_3<733, (outs crrc:$crD), (ins gprc:$RA, gprc:$RB),
                               "efststlt $crD, $RA, $RB", IIC_FPCompare>;
}

// SPE Vector operations

def EVABS          : EVXForm_2<520, (outs gprc:$RT), (ins gprc:$RA),
                               "evabs $RT, $RA", IIC_VecGeneral>;

def EVADDIW        : EVXForm_1<514, (outs gprc:$RT), (ins gprc:$RA, u5imm:$RB),
                               "evaddiw $RT, $RB, $RA", IIC_VecGeneral>;
def EVADDSMIAAW    : EVXForm_2<1225, (outs gprc:$RT), (ins gprc:$RA),
                               "evaddsmiaaw $RT, $RA", IIC_VecComplex>;
def EVADDSSIAAW    : EVXForm_2<1217, (outs gprc:$RT), (ins gprc:$RA),
                               "evaddssiaaw $RT, $RA", IIC_VecComplex>;
def EVADDUSIAAW    : EVXForm_2<1216, (outs gprc:$RT), (ins gprc:$RA),
                               "evaddusiaaw $RT, $RA", IIC_VecComplex>;
def EVADDUMIAAW    : EVXForm_2<1224, (outs gprc:$RT), (ins gprc:$RA),
                               "evaddumiaaw $RT, $RA", IIC_VecComplex>;
def EVADDW         : EVXForm_1<512, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "evaddw $RT, $RA, $RB", IIC_VecGeneral>;

def EVAND          : EVXForm_1<529, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "evand $RT, $RA, $RB", IIC_VecGeneral>;
def EVANDC         : EVXForm_1<530, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "evandc $RT, $RA, $RB", IIC_VecGeneral>;

let isCompare = 1 in {
def EVCMPEQ        : EVXForm_3<564, (outs crrc:$crD), (ins gprc:$RA, gprc:$RB),
                               "evcmpeq $crD, $RA, $RB", IIC_VecGeneral>;
def EVCMPGTS       : EVXForm_3<561, (outs crrc:$crD), (ins gprc:$RA, gprc:$RB),
                               "evcmpgts $crD, $RA, $RB", IIC_VecGeneral>;
def EVCMPGTU       : EVXForm_3<560, (outs crrc:$crD), (ins gprc:$RA, gprc:$RB),
                               "evcmpgtu $crD, $RA, $RB", IIC_VecGeneral>;
def EVCMPLTS       : EVXForm_3<563, (outs crrc:$crD), (ins gprc:$RA, gprc:$RB),
                               "evcmplts $crD, $RA, $RB", IIC_VecGeneral>;
def EVCMPLTU       : EVXForm_3<562, (outs crrc:$crD), (ins gprc:$RA, gprc:$RB),
                               "evcmpltu $crD, $RA, $RB", IIC_VecGeneral>;
}

def EVCNTLSW       : EVXForm_2<526, (outs gprc:$RT), (ins gprc:$RA),
                               "evcntlsw $RT, $RA", IIC_VecGeneral>;
def EVCNTLZW       : EVXForm_2<525, (outs gprc:$RT), (ins gprc:$RA),
                               "evcntlzw $RT, $RA", IIC_VecGeneral>;

def EVDIVWS        : EVXForm_1<1222, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "evdivws $RT, $RA, $RB", IIC_VecComplex>;
def EVDIVWU        : EVXForm_1<1223, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "evdivwu $RT, $RA, $RB", IIC_VecComplex>;

def EVEQV          : EVXForm_1<537, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "eveqv $RT, $RA, $RB", IIC_VecGeneral>;

def EVEXTSB        : EVXForm_2<522, (outs gprc:$RT), (ins gprc:$RA),
                               "evextsb $RT, $RA", IIC_VecGeneral>;
def EVEXTSH        : EVXForm_2<523, (outs gprc:$RT), (ins gprc:$RA),
                               "evextsh $RT, $RA", IIC_VecGeneral>;

def EVFSABS        : EVXForm_2<644, (outs gprc:$RT), (ins gprc:$RA),
                               "evfsabs $RT, $RA", IIC_VecGeneral>;
def EVFSADD        : EVXForm_1<640, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "evfsadd $RT, $RA, $RB", IIC_VecComplex>;
def EVFSCFSF       : EVXForm_2a<659, (outs gprc:$RT), (ins gprc:$RB),
                                "evfscfsf $RT, $RB", IIC_VecComplex>;
def EVFSCFSI       : EVXForm_2a<657, (outs gprc:$RT), (ins gprc:$RB),
                                "evfscfsi $RT, $RB", IIC_VecComplex>;
def EVFSCFUF       : EVXForm_2a<658, (outs gprc:$RT), (ins gprc:$RB),
                                "evfscfuf $RT, $RB", IIC_VecComplex>;
def EVFSCFUI       : EVXForm_2a<650, (outs gprc:$RT), (ins gprc:$RB),
                                "evfscfui $RT, $RB", IIC_VecComplex>;
let isCompare = 1 in {
def EVFSCMPEQ      : EVXForm_3<654, (outs crrc:$crD), (ins gprc:$RA, gprc:$RB),
                               "evfscmpeq $crD, $RA, $RB", IIC_FPSGeneral>;
def EVFSCMPGT      : EVXForm_3<652, (outs crrc:$crD), (ins gprc:$RA, gprc:$RB),
                               "evfscmpgt $crD, $RA, $RB", IIC_FPSGeneral>;
def EVFSCMPLT      : EVXForm_3<653, (outs crrc:$crD), (ins gprc:$RA, gprc:$RB),
                               "evfscmplt $crD, $RA, $RB", IIC_FPSGeneral>;
}

def EVFSCTSF        : EVXForm_2a<663, (outs gprc:$RT), (ins gprc:$RB),
                                "evfsctsf $RT, $RB", IIC_VecComplex>;
def EVFSCTSI        : EVXForm_2a<661, (outs gprc:$RT), (ins gprc:$RB),
                                "evfsctsi $RT, $RB", IIC_VecComplex>;
def EVFSCTSIZ       : EVXForm_2a<666, (outs gprc:$RT), (ins gprc:$RB),
                                "evfsctsiz $RT, $RB", IIC_VecComplex>;
def EVFSCTUF        : EVXForm_2a<662, (outs gprc:$RT), (ins gprc:$RB),
                                "evfsctsf $RT, $RB", IIC_VecComplex>;
def EVFSCTUI        : EVXForm_2a<660, (outs gprc:$RT), (ins gprc:$RB),
                                "evfsctui $RT, $RB", IIC_VecComplex>;
def EVFSCTUIZ       : EVXForm_2a<664, (outs gprc:$RT), (ins gprc:$RB),
                                "evfsctsiz $RT, $RB", IIC_VecComplex>;
def EVFSDIV         : EVXForm_1<649, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "evfsdiv $RT, $RA, $RB", IIC_FPDivD>;
def EVFSMUL         : EVXForm_1<648, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "evfsmul $RT, $RA, $RB", IIC_VecComplex>;
def EVFSNABS        : EVXForm_2<645, (outs gprc:$RT), (ins gprc:$RA),
                               "evfsnabs $RT, $RA", IIC_VecGeneral>;
def EVFSNEG         : EVXForm_2<646, (outs gprc:$RT), (ins gprc:$RA),
                               "evfsneg $RT, $RA", IIC_VecGeneral>;
def EVFSSUB         : EVXForm_1<641, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "evfssub $RT, $RA, $RB", IIC_VecComplex>;

let isCompare = 1 in {
def EVFSTSTEQ       : EVXForm_3<670, (outs crrc:$crD), (ins gprc:$RA, gprc:$RB),
                               "evfststeq $crD, $RA, $RB", IIC_VecGeneral>;
def EVFSTSTGT       : EVXForm_3<668, (outs crrc:$crD), (ins gprc:$RA, gprc:$RB),
                               "evfststgt $crD, $RA, $RB", IIC_VecGeneral>;
def EVFSTSTLT       : EVXForm_3<669, (outs crrc:$crD), (ins gprc:$RA, gprc:$RB),
                               "evfststlt $crD, $RA, $RB", IIC_VecGeneral>;
}

def EVLDD          : EVXForm_D<769, (outs gprc:$RT), (ins spe8dis:$dst),
                               "evldd $RT, $dst", IIC_LdStLoad>;
def EVLDDX         : EVXForm_1<768, (outs gprc:$RT), (ins memrr:$src),
                               "evlddx $RT, $src", IIC_LdStLoad>;
def EVLDH          : EVXForm_D<773, (outs gprc:$RT), (ins spe8dis:$dst),
                               "evldh $RT, $dst", IIC_LdStLoad>;
def EVLDHX         : EVXForm_1<772, (outs gprc:$RT), (ins memrr:$src),
                               "evldhx $RT, $src", IIC_LdStLoad>;
def EVLDW          : EVXForm_D<771, (outs gprc:$RT), (ins spe8dis:$dst),
                               "evldw $RT, $dst", IIC_LdStLoad>;
def EVLDWX         : EVXForm_1<770, (outs gprc:$RT), (ins memrr:$src),
                               "evldwx $RT, $src", IIC_LdStLoad>;
def EVLHHESPLAT    : EVXForm_D<777, (outs gprc:$RT), (ins spe2dis:$dst),
                               "evlhhesplat $RT, $dst", IIC_LdStLoad>;
def EVLHHESPLATX   : EVXForm_1<776, (outs gprc:$RT), (ins memrr:$src),
                               "evlhhesplatx $RT, $src", IIC_LdStLoad>;
def EVLHHOUSPLAT   : EVXForm_D<781, (outs gprc:$RT), (ins spe2dis:$dst),
                               "evlhhousplat $RT, $dst", IIC_LdStLoad>;
def EVLHHOUSPLATX  : EVXForm_1<780, (outs gprc:$RT), (ins memrr:$src),
                               "evlhhousplatx $RT, $src", IIC_LdStLoad>;
def EVLHHOSSPLAT   : EVXForm_D<783, (outs gprc:$RT), (ins spe2dis:$dst),
                               "evlhhossplat $RT, $dst", IIC_LdStLoad>;
def EVLHHOSSPLATX  : EVXForm_1<782, (outs gprc:$RT), (ins memrr:$src),
                               "evlhhossplatx $RT, $src", IIC_LdStLoad>;
def EVLWHE         : EVXForm_D<785, (outs gprc:$RT), (ins spe4dis:$dst),
                               "evlwhe $RT, $dst", IIC_LdStLoad>;
def EVLWHEX        : EVXForm_1<784, (outs gprc:$RT), (ins memrr:$src),
                               "evlwhex $RT, $src", IIC_LdStLoad>;
def EVLWHOS        : EVXForm_D<791, (outs gprc:$RT), (ins spe4dis:$dst),
                               "evlwhos $RT, $dst", IIC_LdStLoad>;
def EVLWHOSX       : EVXForm_1<790, (outs gprc:$RT), (ins memrr:$src),
                               "evlwhosx $RT, $src", IIC_LdStLoad>;
def EVLWHOU        : EVXForm_D<789, (outs gprc:$RT), (ins spe4dis:$dst),
                               "evlwhou $RT, $dst", IIC_LdStLoad>;
def EVLWHOUX       : EVXForm_1<788, (outs gprc:$RT), (ins memrr:$src),
                               "evlwhoux $RT, $src", IIC_LdStLoad>;
def EVLWHSPLAT     : EVXForm_D<797, (outs gprc:$RT), (ins spe4dis:$dst),
                               "evlwhsplat $RT, $dst", IIC_LdStLoad>;
def EVLWHSPLATX    : EVXForm_1<796, (outs gprc:$RT), (ins memrr:$src),
                               "evlwhsplatx $RT, $src", IIC_LdStLoad>;
def EVLWWSPLAT     : EVXForm_D<793, (outs gprc:$RT), (ins spe4dis:$dst),
                               "evlwwsplat $RT, $dst", IIC_LdStLoad>;
def EVLWWSPLATX    : EVXForm_1<792, (outs gprc:$RT), (ins memrr:$src),
                               "evlwwsplatx $RT, $src", IIC_LdStLoad>;

def EVMERGEHI      : EVXForm_1<556, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "evmergehi $RT, $RA, $RB", IIC_VecGeneral>;
def EVMERGELO      : EVXForm_1<557, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "evmergelo $RT, $RA, $RB", IIC_VecGeneral>;
def EVMERGEHILO    : EVXForm_1<558, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "evmergehilo $RT, $RA, $RB", IIC_VecGeneral>;
def EVMERGELOHI    : EVXForm_1<559, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "evmergelohi $RT, $RA, $RB", IIC_VecGeneral>;

def EVMHEGSMFAA    : EVXForm_1<1323, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "evmhegsmfaa $RT, $RA, $RB", IIC_VecComplex>;
def EVMHEGSMFAN    : EVXForm_1<1451, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "evmhegsmfan $RT, $RA, $RB", IIC_VecComplex>;
def EVMHEGSMIAA    : EVXForm_1<1321, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "evmhegsmiaa $RT, $RA, $RB", IIC_VecComplex>;
def EVMHEGSMIAN    : EVXForm_1<1449, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "evmhegsmian $RT, $RA, $RB", IIC_VecComplex>;
def EVMHEGUMIAA    : EVXForm_1<1320, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "evmhegumiaa $RT, $RA, $RB", IIC_VecComplex>;
def EVMHEGUMIAN    : EVXForm_1<1448, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "evmhegumian $RT, $RA, $RB", IIC_VecComplex>;

def EVMHESMF       : EVXForm_1<1035, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "evmhesmf $RT, $RA, $RB", IIC_VecComplex>;
def EVMHESMFA      : EVXForm_1<1067, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "evmhesmfa $RT, $RA, $RB", IIC_VecComplex>;
def EVMHESMFAAW    : EVXForm_1<1291, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "evmhesmfaaw $RT, $RA, $RB", IIC_VecComplex>;
def EVMHESMFANW    : EVXForm_1<1419, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "evmhesmfanw $RT, $RA, $RB", IIC_VecComplex>;
def EVMHESMI       : EVXForm_1<1033, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "evmhesmi $RT, $RA, $RB", IIC_VecComplex>;
def EVMHESMIA      : EVXForm_1<1065, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "evmhesmia $RT, $RA, $RB", IIC_VecComplex>;
def EVMHESMIAAW    : EVXForm_1<1289, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "evmhesmiaaw $RT, $RA, $RB", IIC_VecComplex>;
def EVMHESMIANW    : EVXForm_1<1417, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "evmhesmianw $RT, $RA, $RB", IIC_VecComplex>;
def EVMHESSF       : EVXForm_1<1027, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "evmhessf $RT, $RA, $RB", IIC_VecComplex>;
def EVMHESSFA      : EVXForm_1<1059, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "evmhessfa $RT, $RA, $RB", IIC_VecComplex>;
def EVMHESSFAAW    : EVXForm_1<1283, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "evmhessfaaw $RT, $RA, $RB", IIC_VecComplex>;
def EVMHESSFANW    : EVXForm_1<1411, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "evmhessfanw $RT, $RA, $RB", IIC_VecComplex>;
def EVMHESSIAAW    : EVXForm_1<1281, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "evmhessiaaw $RT, $RA, $RB", IIC_VecComplex>;
def EVMHESSIANW    : EVXForm_1<1409, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "evmhessianw $RT, $RA, $RB", IIC_VecComplex>;
def EVMHEUMI       : EVXForm_1<1032, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "evmheumi $RT, $RA, $RB", IIC_VecComplex>;
def EVMHEUMIA      : EVXForm_1<1064, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "evmheumia $RT, $RA, $RB", IIC_VecComplex>;
def EVMHEUMIAAW    : EVXForm_1<1288, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "evmheumiaaw $RT, $RA, $RB", IIC_VecComplex>;
def EVMHEUMIANW    : EVXForm_1<1416, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "evmheumianw $RT, $RA, $RB", IIC_VecComplex>;
def EVMHEUSIAAW    : EVXForm_1<1280, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "evmheusiaaw $RT, $RA, $RB", IIC_VecComplex>;
def EVMHEUSIANW    : EVXForm_1<1408, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "evmheusianw $RT, $RA, $RB", IIC_VecComplex>;
def EVMHOGSMFAA    : EVXForm_1<1327, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "evmhogsmfaa $RT, $RA, $RB", IIC_VecComplex>;
def EVMHOGSMFAN    : EVXForm_1<1455, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "evmhogsmfan $RT, $RA, $RB", IIC_VecComplex>;
def EVMHOGSMIAA    : EVXForm_1<1325, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "evmhogsmiaa $RT, $RA, $RB", IIC_VecComplex>;
def EVMHOGSMIAN    : EVXForm_1<1453, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "evmhogsmian $RT, $RA, $RB", IIC_VecComplex>;
def EVMHOGUMIAA    : EVXForm_1<1324, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "evmhogumiaa $RT, $RA, $RB", IIC_VecComplex>;
def EVMHOGUMIAN    : EVXForm_1<1452, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "evmhogumian $RT, $RA, $RB", IIC_VecComplex>;
def EVMHOSMF       : EVXForm_1<1039, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "evmhosmf $RT, $RA, $RB", IIC_VecComplex>;
def EVMHOSMFA      : EVXForm_1<1071, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "evmhosmfa $RT, $RA, $RB", IIC_VecComplex>;
def EVMHOSMFAAW    : EVXForm_1<1295, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "evmhosmfaaw $RT, $RA, $RB", IIC_VecComplex>;
def EVMHOSMFANW    : EVXForm_1<1423, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "evmhosmfanw $RT, $RA, $RB", IIC_VecComplex>;
def EVMHOSMI       : EVXForm_1<1037, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "evmhosmi $RT, $RA, $RB", IIC_VecComplex>;
def EVMHOSMIA      : EVXForm_1<1069, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "evmhosmia $RT, $RA, $RB", IIC_VecComplex>;
def EVMHOSMIAAW    : EVXForm_1<1293, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "evmhosmiaaw $RT, $RA, $RB", IIC_VecComplex>;
def EVMHOSMIANW    : EVXForm_1<1421, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "evmhosmianw $RT, $RA, $RB", IIC_VecComplex>;
def EVMHOSSF       : EVXForm_1<1031, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "evmhossf $RT, $RA, $RB", IIC_VecComplex>;
def EVMHOSSFA      : EVXForm_1<1063, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "evmhossfa $RT, $RA, $RB", IIC_VecComplex>;
def EVMHOSSFAAW    : EVXForm_1<1287, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "evmhossfaaw $RT, $RA, $RB", IIC_VecComplex>;
def EVMHOSSFANW    : EVXForm_1<1415, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "evmhossfanw $RT, $RA, $RB", IIC_VecComplex>;
def EVMHOSSIAAW    : EVXForm_1<1285, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "evmhossiaaw $RT, $RA, $RB", IIC_VecComplex>;
def EVMHOSSIANW    : EVXForm_1<1413, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "evmhossianw $RT, $RA, $RB", IIC_VecComplex>;
def EVMHOUMI       : EVXForm_1<1036, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "evmhoumi $RT, $RA, $RB", IIC_VecComplex>;
def EVMHOUMIA      : EVXForm_1<1068, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "evmhoumia $RT, $RA, $RB", IIC_VecComplex>;
def EVMHOUMIAAW    : EVXForm_1<1292, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "evmhoumiaaw $RT, $RA, $RB", IIC_VecComplex>;
def EVMHOUMIANW    : EVXForm_1<1420, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "evmhoumianw $RT, $RA, $RB", IIC_VecComplex>;
def EVMHOUSIAAW    : EVXForm_1<1284, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "evmhousiaaw $RT, $RA, $RB", IIC_VecComplex>;
def EVMHOUSIANW    : EVXForm_1<1412, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "evmhousianw $RT, $RA, $RB", IIC_VecComplex>;

def EVMRA          : EVXForm_2<1220, (outs gprc:$RT), (ins gprc:$RA),
                               "evmra $RT, $RA", IIC_VecComplex>;

def EVMWHSMF       : EVXForm_1<1103, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "evmwhsmf $RT, $RA, $RB", IIC_VecComplex>;
def EVMWHSMFA      : EVXForm_1<1135, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "evmwhsmfa $RT, $RA, $RB", IIC_VecComplex>;
def EVMWHSMI       : EVXForm_1<1101, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "evmwhsmi $RT, $RA, $RB", IIC_VecComplex>;
def EVMWHSMIA      : EVXForm_1<1133, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "evmwhsmia $RT, $RA, $RB", IIC_VecComplex>;
def EVMWHSSF       : EVXForm_1<1095, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "evmwhssf $RT, $RA, $RB", IIC_VecComplex>;
def EVMWHSSFA      : EVXForm_1<1127, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "evmwhssfa $RT, $RA, $RB", IIC_VecComplex>;
def EVMWHUMI       : EVXForm_1<1100, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "evmwhumi $RT, $RA, $RB", IIC_VecComplex>;
def EVMWHUMIA      : EVXForm_1<1132, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "evmwhumia $RT, $RA, $RB", IIC_VecComplex>;
def EVMWLSMIAAW    : EVXForm_1<1353, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "evmwlsmiaaw $RT, $RA, $RB", IIC_VecComplex>;
def EVMWLSMIANW    : EVXForm_1<1481, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "evmwlsmianw $RT, $RA, $RB", IIC_VecComplex>;
def EVMWLSSIAAW    : EVXForm_1<1345, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "evmwlssiaaw $RT, $RA, $RB", IIC_VecComplex>;
def EVMWLSSIANW    : EVXForm_1<1473, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "evmwlssianw $RT, $RA, $RB", IIC_VecComplex>;
def EVMWLUMI       : EVXForm_1<1096, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "evmwlumi $RT, $RA, $RB", IIC_VecComplex>;
def EVMWLUMIA      : EVXForm_1<1128, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "evmwlumia $RT, $RA, $RB", IIC_VecComplex>;
def EVMWLUMIAAW    : EVXForm_1<1352, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "evmwlumiaaw $RT, $RA, $RB", IIC_VecComplex>;
def EVMWLUMIANW    : EVXForm_1<1480, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "evmwlumianw $RT, $RA, $RB", IIC_VecComplex>;
def EVMWLUSIAAW    : EVXForm_1<1344, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "evmwlusiaaw $RT, $RA, $RB", IIC_VecComplex>;
def EVMWLUSIANW    : EVXForm_1<1472, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "evmwlusianw $RT, $RA, $RB", IIC_VecComplex>;
def EVMWSMF        : EVXForm_1<1115, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "evmwsmf $RT, $RA, $RB", IIC_VecComplex>;
def EVMWSMFA       : EVXForm_1<1147, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "evmwsmfa $RT, $RA, $RB", IIC_VecComplex>;
def EVMWSMFAA      : EVXForm_1<1371, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "evmwsmfaa $RT, $RA, $RB", IIC_VecComplex>;
def EVMWSMFAN      : EVXForm_1<1499, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "evmwsmfan $RT, $RA, $RB", IIC_VecComplex>;
def EVMWSMI        : EVXForm_1<1113, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "evmwsmi $RT, $RA, $RB", IIC_VecComplex>;
def EVMWSMIA       : EVXForm_1<1145, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "evmwsmia $RT, $RA, $RB", IIC_VecComplex>;
def EVMWSMIAA      : EVXForm_1<1369, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "evmwsmiaa $RT, $RA, $RB", IIC_VecComplex>;
def EVMWSMIAN      : EVXForm_1<1497, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "evmwsmian $RT, $RA, $RB", IIC_VecComplex>;
def EVMWSSF        : EVXForm_1<1107, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "evmwssf $RT, $RA, $RB", IIC_VecComplex>;
def EVMWSSFA       : EVXForm_1<1139, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "evmwssfa $RT, $RA, $RB", IIC_VecComplex>;
def EVMWSSFAA      : EVXForm_1<1363, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "evmwssfaa $RT, $RA, $RB", IIC_VecComplex>;
def EVMWSSFAN      : EVXForm_1<1491, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "evmwssfan $RT, $RA, $RB", IIC_VecComplex>;
def EVMWUMI        : EVXForm_1<1112, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "evmwumi $RT, $RA, $RB", IIC_VecComplex>;
def EVMWUMIA       : EVXForm_1<1144, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "evmwumia $RT, $RA, $RB", IIC_VecComplex>;
def EVMWUMIAA      : EVXForm_1<1368, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "evmwumiaa $RT, $RA, $RB", IIC_VecComplex>;
def EVMWUMIAN      : EVXForm_1<1496, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "evmwumian $RT, $RA, $RB", IIC_VecComplex>;


def EVNAND         : EVXForm_1<542, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "evnand $RT, $RA, $RB", IIC_VecGeneral>;

def EVNEG          : EVXForm_2<521, (outs gprc:$RT), (ins gprc:$RA),
                               "evneg $RT, $RA", IIC_VecGeneral>;

def EVNOR          : EVXForm_1<536, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "evnor $RT, $RA, $RB", IIC_VecGeneral>;
def EVOR           : EVXForm_1<535, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "evor $RT, $RA, $RB", IIC_VecGeneral>;
def EVORC          : EVXForm_1<539, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "evorc $RT, $RA, $RB", IIC_VecGeneral>;

def EVRLWI         : EVXForm_1<554, (outs gprc:$RT), (ins gprc:$RA, u5imm:$RB),
                               "evrlwi $RT, $RA, $RB", IIC_VecGeneral>;
def EVRLW          : EVXForm_1<552, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "evrlw $RT, $RA, $RB", IIC_VecGeneral>;

def EVRNDW         : EVXForm_2<524, (outs gprc:$RT), (ins gprc:$RA),
                               "evrndw $RT, $RA", IIC_VecGeneral>;

def EVSEL          : EVXForm_4<79, (outs gprc:$RT),
                               (ins gprc:$RA, gprc:$RB, crrc:$crD),
                               "evsel crD,$RT,$RA,$RB", IIC_VecGeneral>;

def EVSLWI         : EVXForm_1<550, (outs gprc:$RT), (ins gprc:$RA, u5imm:$RB),
                               "evslwi $RT, $RA, $RB", IIC_VecGeneral>;
def EVSLW          : EVXForm_1<548, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "evslw $RT, $RA, $RB", IIC_VecGeneral>;

def EVSPLATFI      : EVXForm_2<555, (outs gprc:$RT), (ins s5imm:$RA),
                               "evsplatfi $RT, $RA", IIC_VecGeneral>;
def EVSPLATI       : EVXForm_2<553, (outs gprc:$RT), (ins s5imm:$RA),
                               "evsplati $RT, $RA", IIC_VecGeneral>;

def EVSRWIS        : EVXForm_1<547, (outs gprc:$RT), (ins gprc:$RA, u5imm:$RB),
                               "evsrwis $RT, $RA, $RB", IIC_VecGeneral>;
def EVSRWIU        : EVXForm_1<546, (outs gprc:$RT), (ins gprc:$RA, u5imm:$RB),
                               "evsrwiu $RT, $RA, $RB", IIC_VecGeneral>;
def EVSRWS         : EVXForm_1<545, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "evsrws $RT, $RA, $RB", IIC_VecGeneral>;
def EVSRWU         : EVXForm_1<544, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "evsrwu $RT, $RA, $RB", IIC_VecGeneral>;

def EVSTDD         : EVXForm_D<801, (outs), (ins gprc:$RT, spe8dis:$dst),
                               "evstdd $RT, $dst", IIC_LdStStore>;
def EVSTDDX        : EVXForm_1<800, (outs), (ins gprc:$RT, memrr:$dst),
                               "evstddx $RT, $dst", IIC_LdStStore>;
def EVSTDH         : EVXForm_D<805, (outs), (ins gprc:$RT, spe8dis:$dst),
                               "evstdh $RT, $dst", IIC_LdStStore>;
def EVSTDHX        : EVXForm_1<804, (outs), (ins gprc:$RT, memrr:$dst),
                               "evstdhx $RT, $dst", IIC_LdStStore>;
def EVSTDW         : EVXForm_D<803, (outs), (ins gprc:$RT, spe8dis:$dst),
                               "evstdw $RT, $dst", IIC_LdStStore>;
def EVSTDWX        : EVXForm_1<802, (outs), (ins gprc:$RT, memrr:$dst),
                               "evstdwx $RT, $dst", IIC_LdStStore>;
def EVSTWHE        : EVXForm_D<817, (outs), (ins gprc:$RT, spe4dis:$dst),
                               "evstwhe $RT, $dst", IIC_LdStStore>;
def EVSTWHEX       : EVXForm_1<816, (outs), (ins gprc:$RT, memrr:$dst),
                               "evstwhex $RT, $dst", IIC_LdStStore>;
def EVSTWHO        : EVXForm_D<821, (outs), (ins gprc:$RT, spe4dis:$dst),
                               "evstwho $RT, $dst", IIC_LdStStore>;
def EVSTWHOX       : EVXForm_1<820, (outs), (ins gprc:$RT, memrr:$dst),
                               "evstwhox $RT, $dst", IIC_LdStStore>;
def EVSTWWE        : EVXForm_D<825, (outs), (ins gprc:$RT, spe4dis:$dst),
                               "evstwwe $RT, $dst", IIC_LdStStore>;
def EVSTWWEX       : EVXForm_1<824, (outs), (ins gprc:$RT, memrr:$dst),
                               "evstwwex $RT, $dst", IIC_LdStStore>;
def EVSTWWO        : EVXForm_D<829, (outs), (ins gprc:$RT, spe4dis:$dst),
                               "evstwwo $RT, $dst", IIC_LdStStore>;
def EVSTWWOX       : EVXForm_1<828, (outs), (ins gprc:$RT, memrr:$dst),
                               "evstwwox $RT, $dst", IIC_LdStStore>;

def EVSUBFSSIAAW   : EVXForm_2<1219, (outs gprc:$RT), (ins gprc:$RA),
                               "evsubfssiaaw $RT, $RA", IIC_VecComplex>;
def EVSUBFSMIAAW   : EVXForm_2<1227, (outs gprc:$RT), (ins gprc:$RA),
                               "evsubfsmiaaw $RT, $RA", IIC_VecComplex>;
def EVSUBFUMIAAW   : EVXForm_2<1226, (outs gprc:$RT), (ins gprc:$RA),
                               "evsubfumiaaw $RT, $RA", IIC_VecComplex>;
def EVSUBFUSIAAW   : EVXForm_2<1218, (outs gprc:$RT), (ins gprc:$RA),
                               "evsubfusiaaw $RT, $RA", IIC_VecComplex>;
def EVSUBFW        : EVXForm_1<516, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "evsubfw $RT, $RA, $RB", IIC_VecGeneral>;
def EVSUBIFW       : EVXForm_1<518, (outs gprc:$RT), (ins u5imm:$RA, gprc:$RB),
                               "evsubifw $RT, $RA, $RB", IIC_VecGeneral>;
def EVXOR          : EVXForm_1<534, (outs gprc:$RT), (ins gprc:$RA, gprc:$RB),
                               "evxor $RT, $RA, $RB", IIC_VecGeneral>;

} // HasSPE
