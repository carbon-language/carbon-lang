//===- MBlazeInstrFSL.td - MBlaze FSL Instruction defs -----*- tablegen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// FSL Instruction Formats
//===----------------------------------------------------------------------===//
class FSLGetD<bits<6> op, bits<11> flags, string instr_asm, Intrinsic OpNode> :
              TA<op, flags, (outs CPURegs:$dst), (ins CPURegs:$b),
                 !strconcat(instr_asm, " $dst, $b"),
                 [(set CPURegs:$dst, (OpNode CPURegs:$b))], IIAlu>;

class FSLGet<bits<6> op, string instr_asm, Intrinsic OpNode> :
             TAI<op, (outs CPURegs:$dst), (ins fslimm:$b),
                 !strconcat(instr_asm, " $dst, $b"),
                 [(set CPURegs:$dst, (OpNode immZExt4:$b))], IIAlu>;

class FSLPutD<bits<6> op, bits<11> flags, string instr_asm, Intrinsic OpNode> :
              TA<op, flags, (outs), (ins CPURegs:$v, CPURegs:$b),
                 !strconcat(instr_asm, " $v, $b"),
                 [(OpNode CPURegs:$v, CPURegs:$b)], IIAlu>;

class FSLPut<bits<6> op, string instr_asm, Intrinsic OpNode> :
             TAI<op, (outs), (ins CPURegs:$v, fslimm:$b),
                 !strconcat(instr_asm, " $v, $b"),
                 [(OpNode CPURegs:$v, immZExt4:$b)], IIAlu>;

class FSLPutTD<bits<6> op, bits<11> flags, string instr_asm, Intrinsic OpNode> :
               TA<op, flags, (outs), (ins CPURegs:$b),
                  !strconcat(instr_asm, " $b"),
                  [(OpNode CPURegs:$b)], IIAlu>;

class FSLPutT<bits<6> op, string instr_asm, Intrinsic OpNode> :
              TAI<op, (outs), (ins fslimm:$b),
                  !strconcat(instr_asm, " $b"),
                  [(OpNode immZExt4:$b)], IIAlu>;

//===----------------------------------------------------------------------===//
// FSL Get Instructions
//===----------------------------------------------------------------------===//
def GET      : FSLGet<0x1B, "get      ", int_mblaze_fsl_get>;
def AGET     : FSLGet<0x1B, "aget     ", int_mblaze_fsl_aget>;
def CGET     : FSLGet<0x1B, "cget     ", int_mblaze_fsl_cget>;
def CAGET    : FSLGet<0x1B, "caget    ", int_mblaze_fsl_caget>;
def EGET     : FSLGet<0x1B, "eget     ", int_mblaze_fsl_eget>;
def EAGET    : FSLGet<0x1B, "eaget    ", int_mblaze_fsl_eaget>;
def ECGET    : FSLGet<0x1B, "ecget    ", int_mblaze_fsl_ecget>;
def ECAGET   : FSLGet<0x1B, "ecaget   ", int_mblaze_fsl_ecaget>;
def NGET     : FSLGet<0x1B, "nget     ", int_mblaze_fsl_nget>;
def NAGET    : FSLGet<0x1B, "naget    ", int_mblaze_fsl_naget>;
def NCGET    : FSLGet<0x1B, "ncget    ", int_mblaze_fsl_ncget>;
def NCAGET   : FSLGet<0x1B, "ncaget   ", int_mblaze_fsl_ncaget>;
def NEGET    : FSLGet<0x1B, "neget    ", int_mblaze_fsl_neget>;
def NEAGET   : FSLGet<0x1B, "neaget   ", int_mblaze_fsl_neaget>;
def NECGET   : FSLGet<0x1B, "necget   ", int_mblaze_fsl_necget>;
def NECAGET  : FSLGet<0x1B, "necaget  ", int_mblaze_fsl_necaget>;
def TGET     : FSLGet<0x1B, "tget     ", int_mblaze_fsl_tget>;
def TAGET    : FSLGet<0x1B, "taget    ", int_mblaze_fsl_taget>;
def TCGET    : FSLGet<0x1B, "tcget    ", int_mblaze_fsl_tcget>;
def TCAGET   : FSLGet<0x1B, "tcaget   ", int_mblaze_fsl_tcaget>;
def TEGET    : FSLGet<0x1B, "teget    ", int_mblaze_fsl_teget>;
def TEAGET   : FSLGet<0x1B, "teaget   ", int_mblaze_fsl_teaget>;
def TECGET   : FSLGet<0x1B, "tecget   ", int_mblaze_fsl_tecget>;
def TECAGET  : FSLGet<0x1B, "tecaget  ", int_mblaze_fsl_tecaget>;
def TNGET    : FSLGet<0x1B, "tnget    ", int_mblaze_fsl_tnget>;
def TNAGET   : FSLGet<0x1B, "tnaget   ", int_mblaze_fsl_tnaget>;
def TNCGET   : FSLGet<0x1B, "tncget   ", int_mblaze_fsl_tncget>;
def TNCAGET  : FSLGet<0x1B, "tncaget  ", int_mblaze_fsl_tncaget>;
def TNEGET   : FSLGet<0x1B, "tneget   ", int_mblaze_fsl_tneget>;
def TNEAGET  : FSLGet<0x1B, "tneaget  ", int_mblaze_fsl_tneaget>;
def TNECGET  : FSLGet<0x1B, "tnecget  ", int_mblaze_fsl_tnecget>;
def TNECAGET : FSLGet<0x1B, "tnecaget ", int_mblaze_fsl_tnecaget>;

//===----------------------------------------------------------------------===//
// FSL Dynamic Get Instructions
//===----------------------------------------------------------------------===//
def GETD      : FSLGetD<0x1B, 0x00, "getd     ", int_mblaze_fsl_get>;
def AGETD     : FSLGetD<0x1B, 0x00, "agetd    ", int_mblaze_fsl_aget>;
def CGETD     : FSLGetD<0x1B, 0x00, "cgetd    ", int_mblaze_fsl_cget>;
def CAGETD    : FSLGetD<0x1B, 0x00, "cagetd   ", int_mblaze_fsl_caget>;
def EGETD     : FSLGetD<0x1B, 0x00, "egetd    ", int_mblaze_fsl_eget>;
def EAGETD    : FSLGetD<0x1B, 0x00, "eagetd   ", int_mblaze_fsl_eaget>;
def ECGETD    : FSLGetD<0x1B, 0x00, "ecgetd   ", int_mblaze_fsl_ecget>;
def ECAGETD   : FSLGetD<0x1B, 0x00, "ecagetd  ", int_mblaze_fsl_ecaget>;
def NGETD     : FSLGetD<0x1B, 0x00, "ngetd    ", int_mblaze_fsl_nget>;
def NAGETD    : FSLGetD<0x1B, 0x00, "nagetd   ", int_mblaze_fsl_naget>;
def NCGETD    : FSLGetD<0x1B, 0x00, "ncgetd   ", int_mblaze_fsl_ncget>;
def NCAGETD   : FSLGetD<0x1B, 0x00, "ncagetd  ", int_mblaze_fsl_ncaget>;
def NEGETD    : FSLGetD<0x1B, 0x00, "negetd   ", int_mblaze_fsl_neget>;
def NEAGETD   : FSLGetD<0x1B, 0x00, "neagetd  ", int_mblaze_fsl_neaget>;
def NECGETD   : FSLGetD<0x1B, 0x00, "necgetd  ", int_mblaze_fsl_necget>;
def NECAGETD  : FSLGetD<0x1B, 0x00, "necagetd ", int_mblaze_fsl_necaget>;
def TGETD     : FSLGetD<0x1B, 0x00, "tgetd    ", int_mblaze_fsl_tget>;
def TAGETD    : FSLGetD<0x1B, 0x00, "tagetd   ", int_mblaze_fsl_taget>;
def TCGETD    : FSLGetD<0x1B, 0x00, "tcgetd   ", int_mblaze_fsl_tcget>;
def TCAGETD   : FSLGetD<0x1B, 0x00, "tcagetd  ", int_mblaze_fsl_tcaget>;
def TEGETD    : FSLGetD<0x1B, 0x00, "tegetd   ", int_mblaze_fsl_teget>;
def TEAGETD   : FSLGetD<0x1B, 0x00, "teagetd  ", int_mblaze_fsl_teaget>;
def TECGETD   : FSLGetD<0x1B, 0x00, "tecgetd  ", int_mblaze_fsl_tecget>;
def TECAGETD  : FSLGetD<0x1B, 0x00, "tecagetd ", int_mblaze_fsl_tecaget>;
def TNGETD    : FSLGetD<0x1B, 0x00, "tngetd   ", int_mblaze_fsl_tnget>;
def TNAGETD   : FSLGetD<0x1B, 0x00, "tnagetd  ", int_mblaze_fsl_tnaget>;
def TNCGETD   : FSLGetD<0x1B, 0x00, "tncgetd  ", int_mblaze_fsl_tncget>;
def TNCAGETD  : FSLGetD<0x1B, 0x00, "tncagetd ", int_mblaze_fsl_tncaget>;
def TNEGETD   : FSLGetD<0x1B, 0x00, "tnegetd  ", int_mblaze_fsl_tneget>;
def TNEAGETD  : FSLGetD<0x1B, 0x00, "tneagetd ", int_mblaze_fsl_tneaget>;
def TNECGETD  : FSLGetD<0x1B, 0x00, "tnecgetd ", int_mblaze_fsl_tnecget>;
def TNECAGETD : FSLGetD<0x1B, 0x00, "tnecagetd", int_mblaze_fsl_tnecaget>;

//===----------------------------------------------------------------------===//
// FSL Put Instructions
//===----------------------------------------------------------------------===//
def PUT     :  FSLPut<0x1B, "put      ", int_mblaze_fsl_put>;
def APUT    :  FSLPut<0x1B, "aput     ", int_mblaze_fsl_aput>;
def CPUT    :  FSLPut<0x1B, "cput     ", int_mblaze_fsl_cput>;
def CAPUT   :  FSLPut<0x1B, "caput    ", int_mblaze_fsl_caput>;
def NPUT    :  FSLPut<0x1B, "nput     ", int_mblaze_fsl_nput>;
def NAPUT   :  FSLPut<0x1B, "naput    ", int_mblaze_fsl_naput>;
def NCPUT   :  FSLPut<0x1B, "ncput    ", int_mblaze_fsl_ncput>;
def NCAPUT  :  FSLPut<0x1B, "ncaput   ", int_mblaze_fsl_ncaput>;
def TPUT    : FSLPutT<0x1B, "tput     ", int_mblaze_fsl_tput>;
def TAPUT   : FSLPutT<0x1B, "taput    ", int_mblaze_fsl_taput>;
def TCPUT   : FSLPutT<0x1B, "tcput    ", int_mblaze_fsl_tcput>;
def TCAPUT  : FSLPutT<0x1B, "tcaput   ", int_mblaze_fsl_tcaput>;
def TNPUT   : FSLPutT<0x1B, "tnput    ", int_mblaze_fsl_tnput>;
def TNAPUT  : FSLPutT<0x1B, "tnaput   ", int_mblaze_fsl_tnaput>;
def TNCPUT  : FSLPutT<0x1B, "tncput   ", int_mblaze_fsl_tncput>;
def TNCAPUT : FSLPutT<0x1B, "tncaput  ", int_mblaze_fsl_tncaput>;

//===----------------------------------------------------------------------===//
// FSL Dynamic Put Instructions
//===----------------------------------------------------------------------===//
def PUTD     :  FSLPutD<0x1B, 0x00, "putd     ", int_mblaze_fsl_put>;
def APUTD    :  FSLPutD<0x1B, 0x00, "aputd    ", int_mblaze_fsl_aput>;
def CPUTD    :  FSLPutD<0x1B, 0x00, "cputd    ", int_mblaze_fsl_cput>;
def CAPUTD   :  FSLPutD<0x1B, 0x00, "caputd   ", int_mblaze_fsl_caput>;
def NPUTD    :  FSLPutD<0x1B, 0x00, "nputd    ", int_mblaze_fsl_nput>;
def NAPUTD   :  FSLPutD<0x1B, 0x00, "naputd   ", int_mblaze_fsl_naput>;
def NCPUTD   :  FSLPutD<0x1B, 0x00, "ncputd   ", int_mblaze_fsl_ncput>;
def NCAPUTD  :  FSLPutD<0x1B, 0x00, "ncaputd  ", int_mblaze_fsl_ncaput>;
def TPUTD    : FSLPutTD<0x1B, 0x00, "tputd    ", int_mblaze_fsl_tput>;
def TAPUTD   : FSLPutTD<0x1B, 0x00, "taputd   ", int_mblaze_fsl_taput>;
def TCPUTD   : FSLPutTD<0x1B, 0x00, "tcputd   ", int_mblaze_fsl_tcput>;
def TCAPUTD  : FSLPutTD<0x1B, 0x00, "tcaputd  ", int_mblaze_fsl_tcaput>;
def TNPUTD   : FSLPutTD<0x1B, 0x00, "tnputd   ", int_mblaze_fsl_tnput>;
def TNAPUTD  : FSLPutTD<0x1B, 0x00, "tnaputd  ", int_mblaze_fsl_tnaput>;
def TNCPUTD  : FSLPutTD<0x1B, 0x00, "tncputd  ", int_mblaze_fsl_tncput>;
def TNCAPUTD : FSLPutTD<0x1B, 0x00, "tncaputd ", int_mblaze_fsl_tncaput>;
