
let Namespace = "X86" in {

// def PHI              :  TargInstr<"PHI", 2, [MBI_Reg, MBI_Reg]>;
// def NOOP             :  TargInstr<"NOOP", 2, [MBI_Reg, MBI_Reg]>;
// def ADJCALLSTACKDOWN :  TargInstr<"ADJCALLSTACKDOWN", 2, [MBI_Reg, MBI_Reg]>;
// def ADJCALLSTACKUP   :  TargInstr<"ADJCALLSTACKUP", 2, [MBI_Reg, MBI_Reg]>;
def IMPLICIT_USE     :  TargInstr<"IMPLICIT_USE", 2, [MBI_Reg, MBI_Reg]>;
def IMPLICIT_DEF     :  TargInstr<"IMPLICIT_DEF", 0, []>;

def RET      :  TargInstr<"RET", 0, []>;
def JMP      :  TargInstr<"JMP", 1, [MBI_PCDisp]>;
// def JB       :  TargInstr<"JB", 2, [MBI_Reg, MBI_Reg]>;
// def JAE      :  TargInstr<"JAE", 2, [MBI_Reg, MBI_Reg]>;
// def JE       :  TargInstr<"JE", 2, [MBI_Reg, MBI_Reg]>;
// def JNE      :  TargInstr<"JNE", 2, [MBI_Reg, MBI_Reg]>;
// def JBE      :  TargInstr<"JBE", 2, [MBI_Reg, MBI_Reg]>;
// def JA       :  TargInstr<"JA", 2, [MBI_Reg, MBI_Reg]>;
// def JS       :  TargInstr<"JS", 2, [MBI_Reg, MBI_Reg]>;
// def JNS      :  TargInstr<"JNS", 2, [MBI_Reg, MBI_Reg]>;
// def JL       :  TargInstr<"JL", 2, [MBI_Reg, MBI_Reg]>;
// def JGE      :  TargInstr<"JGE", 2, [MBI_Reg, MBI_Reg]>;
// def JLE      :  TargInstr<"JLE", 2, [MBI_Reg, MBI_Reg]>;
// def JG       :  TargInstr<"JG", 2, [MBI_Reg, MBI_Reg]>;
// def LEAVE    :  TargInstr<"LEAVE", 2, [MBI_Reg, MBI_Reg]>;
// def BSWAPr32 :  TargInstr<"BSWAPr32", 2, [MBI_Reg, MBI_Reg]>;
// def XCHGrr8  :  TargInstr<"XCHGrr8", 2, [MBI_Reg, MBI_Reg]>;
// def XCHGrr16 :  TargInstr<"XCHGrr16", 2, [MBI_Reg, MBI_Reg]>;
// def XCHGrr32 :  TargInstr<"XCHGrr32", 2, [MBI_Reg, MBI_Reg]>;
// def LEAr16   :  TargInstr<"LEAr16", 2, [MBI_Reg, MBI_Reg]>;
// def LEAr32   :  TargInstr<"LEAr32", 2, [MBI_Reg, MBI_Reg]>;
def MOVrr8   :  TargInstr<"MOV8rr", 1, [MBI_Reg]>;
def MOVrr16  :  TargInstr<"MOV16rr", 1, [MBI_Reg]>;
def MOVrr32  :  TargInstr<"MOV32rr", 1, [MBI_Reg]>;
def MOVir8   :  TargInstr<"MOV8ir", 1, [MBI_ZImm]>;
def MOVir16  :  TargInstr<"MOV16ir", 1, [MBI_ZImm]>;
def MOVir32  :  TargInstr<"MOV32ir", 1, [MBI_ZImm]>;
// def MOVim8   :  TargInstr<"MOVim8", 2, [MBI_Reg, MBI_ZImm]>;
// def MOVim16  :  TargInstr<"MOVim16", 2, [MBI_Reg, MBI_ZImm]>;
// def MOVim32  :  TargInstr<"MOVim32", 2, [MBI_Reg, MBI_ZImm]>;
def MOVmr8   :  TargInstr<"MOV8mr", 4, [MBI_FrameRef]>;
def MOVmr16  :  TargInstr<"MOV16mr", 4, [MBI_FrameRef]>;
def MOVmr32  :  TargInstr<"MOV32mr", 4, [MBI_FrameRef]>;
def MOVrm8   :  TargInstr<"MOV8rm", 5, [MBI_FrameRef, MBI_Reg]>; // 5 Operands?, stores to NULL
def MOVrm16  :  TargInstr<"MOV16rm", 5, [MBI_FrameRef, MBI_Reg]>; // 5 Operands?, stores to NULL
def MOVrm32  :  TargInstr<"MOV32rm", 5, [MBI_FrameRef, MBI_Reg]>; // 5 Operands?, stores to NULL
def MULr8    :  TargInstr<"MUL8r", 1, [MBI_Reg]>;
// def MULr16   :  TargInstr<"MULr16", 2, [MBI_Reg, MBI_Reg]>;
// def MULr32   :  TargInstr<"MULr32", 2, [MBI_Reg, MBI_Reg]>;
def DIVr8    :  TargInstr<"DIV8r", 1, [MBI_Reg]>; // "DestReg" as arg, "1" as the dest
def DIVr16   :  TargInstr<"DIV16r", 1, [MBI_Reg]>;
def DIVr32   :  TargInstr<"DIV32r", 1, [MBI_Reg]>;
def IDIVr8   :  TargInstr<"IDIV8r", 1, [MBI_Reg]>;
def IDIVr16  : TargInstr<"IDIV16r", 1, [MBI_Reg]>;
def IDIVr32  : TargInstr<"IDIV32r", 1, [MBI_Reg]>;
// def CBW      :  TargInstr<"CBW", 2, [MBI_Reg, MBI_Reg]>;
// def CWD      :  TargInstr<"CWD", 2, [MBI_Reg, MBI_Reg]>;
// def CDQ      :  TargInstr<"CDQ", 2, [MBI_Reg, MBI_Reg]>;

def NEGr8  :  TargInstr<"NEG8r", 1, [MBI_Reg]>;
def NEGr16 :  TargInstr<"NEG16r", 1, [MBI_Reg]>;
def NEGr32 :  TargInstr<"NEG32r", 1, [MBI_Reg]>;
def NOTr8  :  TargInstr<"NOT8r", 1, [MBI_Reg]>;
def NOTr16 :  TargInstr<"NOT16r", 1, [MBI_Reg]>;
def NOTr32 :  TargInstr<"NOT32r", 1, [MBI_Reg]>;
def INCr8  :  TargInstr<"INC8r", 1, [MBI_Reg]>;
def INCr16 :  TargInstr<"INC16r", 1, [MBI_Reg]>;
def INCr32 :  TargInstr<"INC32r", 1, [MBI_Reg]>;
def DECr8  :  TargInstr<"DEC8r", 1, [MBI_Reg]>;
def DECr16 :  TargInstr<"DEC16r", 1, [MBI_Reg]>;
def DECr32 :  TargInstr<"DEC32r", 1, [MBI_Reg]>;

def ADDrr8    :  TargInstr<"ADD8rr", 2, [MBI_Reg, MBI_Reg]>;
def ADDrr16   :  TargInstr<"ADD16rr", 2, [MBI_Reg, MBI_Reg]>;
def ADDrr32   :  TargInstr<"ADD32rr", 2, [MBI_Reg, MBI_Reg]>;
def ADDri8    :  TargInstr<"ADD8ri", 2, [MBI_Reg, MBI_ZImm]>;
def ADDri16   :  TargInstr<"ADD16ri", 2, [MBI_Reg, MBI_ZImm]>;
def ADDri32   :  TargInstr<"ADD32ri", 2, [MBI_Reg, MBI_ZImm]>;
def ADDri16b  :  TargInstr<"ADD16bri", 2, [MBI_Reg, MBI_ZImm]>;
def ADDri32b  :  TargInstr<"ADD32bri", 2, [MBI_Reg, MBI_ZImm]>;
def ADCrr32   :  TargInstr<"ADC32rr", 2, [MBI_Reg, MBI_Reg]>;
def SUBrr8    :  TargInstr<"SUB8rr", 2, [MBI_Reg, MBI_Reg]>;
def SUBrr16   :  TargInstr<"SUB16rr", 2, [MBI_Reg, MBI_Reg]>;
def SUBrr32   :  TargInstr<"SUB32rr", 2, [MBI_Reg, MBI_Reg]>;
def SUBri8    :  TargInstr<"SUB8ri", 2, [MBI_Reg, MBI_ZImm]>;
def SUBri16   :  TargInstr<"SUB16ri", 2, [MBI_Reg, MBI_ZImm]>;
def SUBri32   :  TargInstr<"SUB32ri", 2, [MBI_Reg, MBI_ZImm]>;
def SUBri16b  :  TargInstr<"SUB16bri", 2, [MBI_Reg, MBI_ZImm]>;
def SUBri32b  :  TargInstr<"SUB32bri", 2, [MBI_Reg, MBI_ZImm]>;
def SBBrr32   :  TargInstr<"SBB32rr", 2, [MBI_Reg, MBI_Reg]>;
def IMULrr16  :  TargInstr<"IMUL16rr", 2, [MBI_Reg, MBI_Reg]>;
def IMULrr32  :  TargInstr<"IMUL32rr", 2, [MBI_Reg, MBI_Reg]>;
def IMULri16  :  TargInstr<"IMUL16ri", 2, [MBI_Reg, MBI_ZImm]>;
def IMULri32  :  TargInstr<"IMUL32ri", 2, [MBI_Reg, MBI_ZImm]>;
def IMULri16b :  TargInstr<"IMUL16bri", 2, [MBI_Reg, MBI_ZImm]>;
def IMULri32b :  TargInstr<"IMUL32bri", 2, [MBI_Reg, MBI_ZImm]>;

def ANDrr8   :  TargInstr<"AND8rr", 2, [MBI_Reg, MBI_Reg]>;
def ANDrr16  :  TargInstr<"AND16rr", 2, [MBI_Reg, MBI_Reg]>;
def ANDrr32  :  TargInstr<"AND32rr", 2, [MBI_Reg, MBI_Reg]>;
def ANDri8   :  TargInstr<"AND8ri", 2, [MBI_Reg, MBI_ZImm]>;
def ANDri16  :  TargInstr<"AND16ri", 2, [MBI_Reg, MBI_ZImm]>;
def ANDri32  :  TargInstr<"AND32ri", 2, [MBI_Reg, MBI_ZImm]>;
def ANDri16b :  TargInstr<"AND16bri", 2, [MBI_Reg, MBI_ZImm]>;
def ANDri32b :  TargInstr<"AND32bri", 2, [MBI_Reg, MBI_ZImm]>;
def ORrr8    :  TargInstr<"OR8rr", 2, [MBI_Reg, MBI_Reg]>;
def ORrr16   :  TargInstr<"OR16rr", 2, [MBI_Reg, MBI_Reg]>;
def ORrr32   :  TargInstr<"OR32rr", 2, [MBI_Reg, MBI_Reg]>;
def ORri8    :  TargInstr<"OR8ri", 2, [MBI_Reg, MBI_ZImm]>;
def ORri16   :  TargInstr<"OR16ri", 2, [MBI_Reg, MBI_ZImm]>;
def ORri32   :  TargInstr<"OR32ri", 2, [MBI_Reg, MBI_ZImm]>;
def ORri16b  :  TargInstr<"OR16bri", 2, [MBI_Reg, MBI_ZImm]>;
def ORri32b  :  TargInstr<"OR32bri", 2, [MBI_Reg, MBI_ZImm]>;
def XORrr8   :  TargInstr<"XOR8rr", 2, [MBI_Reg, MBI_Reg]>;
def XORrr16  :  TargInstr<"XOR16rr", 2, [MBI_Reg, MBI_Reg]>;
def XORrr32  :  TargInstr<"XOR32rr", 2, [MBI_Reg, MBI_Reg]>;
def XORri8   :  TargInstr<"XOR8ri", 2, [MBI_Reg, MBI_ZImm]>;
def XORri16  :  TargInstr<"XOR16ri", 2, [MBI_Reg, MBI_ZImm]>;
def XORri32  :  TargInstr<"XOR32ri", 2, [MBI_Reg, MBI_ZImm]>;
def XORri16b :  TargInstr<"XOR16bri", 2, [MBI_Reg, MBI_ZImm]>;
def XORri32b :  TargInstr<"XOR32bri", 2, [MBI_Reg, MBI_ZImm]>;

def TESTrr8  :  TargInstr<"TEST8rr", 2, [MBI_Reg, MBI_Reg]>;
def TESTrr16 :  TargInstr<"TEST16rr", 2, [MBI_Reg, MBI_Reg]>;
def TESTrr32 :  TargInstr<"TEST32rr", 2, [MBI_Reg, MBI_Reg]>;
def TESTri8  :  TargInstr<"TEST8ri", 2, [MBI_Reg, MBI_ZImm]>;
def TESTri16 :  TargInstr<"TEST16ri", 2, [MBI_Reg, MBI_ZImm]>;
def TESTri32 :  TargInstr<"TEST32ri", 2, [MBI_Reg, MBI_ZImm]>;

def SHLrr8  :  TargInstr<"SHL8rr", 1, [MBI_Reg]>;
def SHLrr16 :  TargInstr<"SHL16rr", 1, [MBI_Reg]>;
def SHLrr32 :  TargInstr<"SHL32rr", 1, [MBI_Reg]>;
def SHLir8  :  TargInstr<"SHL8ir", 2, [MBI_Reg, MBI_ZImm]>;
def SHLir16 :  TargInstr<"SHL16ir", 2, [MBI_Reg, MBI_ZImm]>;
def SHLir32 :  TargInstr<"SHL32ir", 2, [MBI_Reg, MBI_ZImm]>;
def SHRrr8  :  TargInstr<"SHR8rr", 1, [MBI_Reg]>;
def SHRrr16 :  TargInstr<"SHR16rr", 1, [MBI_Reg]>;
def SHRrr32 :  TargInstr<"SHR32rr", 1, [MBI_Reg]>;
def SHRir8  :  TargInstr<"SHR8ir", 2, [MBI_Reg, MBI_ZImm]>;
def SHRir16 :  TargInstr<"SHR16ir", 2, [MBI_Reg, MBI_ZImm]>;
def SHRir32 :  TargInstr<"SHR32ir", 2, [MBI_Reg, MBI_ZImm]>;
def SARrr8  :  TargInstr<"SAR8rr", 1, [MBI_Reg]>;
def SARrr16 :  TargInstr<"SAR16rr", 1, [MBI_Reg]>;
def SARrr32 :  TargInstr<"SAR32rr", 1, [MBI_Reg]>;
def SARir8  :  TargInstr<"SAR8ir", 2, [MBI_Reg, MBI_ZImm]>;
def SARir16 :  TargInstr<"SAR16ir", 2, [MBI_Reg, MBI_ZImm]>;
def SARir32 :  TargInstr<"SAR32ir", 2, [MBI_Reg, MBI_ZImm]>;

def SHLDrr32 :  TargInstr<"SHLD32rr", 2, [MBI_Reg, MBI_Reg]>;
def SHLDir32 :  TargInstr<"SHLD32ir", 3, [MBI_Reg, MBI_Reg, MBI_ZImm]>;
def SHRDrr32 :  TargInstr<"SHRD32rr", 2, [MBI_Reg, MBI_Reg]>;
def SHRDir32 :  TargInstr<"SHRD32ir", 3, [MBI_Reg, MBI_Reg, MBI_ZImm]>;

def SAHF   :  TargInstr<"SAHF", 0, []>; // store in "1" ?
// def SETBr  :  TargInstr<"SETBr", 2, [MBI_Reg, MBI_Reg]>;
// def SETAEr :  TargInstr<"SETAEr", 2, [MBI_Reg, MBI_Reg]>;
// def SETEr  :  TargInstr<"SETEr", 2, [MBI_Reg, MBI_Reg]>;
def SETNEr :  TargInstr<"SETNEr", 0, []>;
// def SETBEr :  TargInstr<"SETBEr", 2, [MBI_Reg, MBI_Reg]>;
// def SETAr  :  TargInstr<"SETAr", 2, [MBI_Reg, MBI_Reg]>;
// def SETSr  :  TargInstr<"SETSr", 2, [MBI_Reg, MBI_Reg]>;
// def SETNSr :  TargInstr<"SETNSr", 2, [MBI_Reg, MBI_Reg]>;
// def SETLr  :  TargInstr<"SETLr", 2, [MBI_Reg, MBI_Reg]>;
// def SETGEr :  TargInstr<"SETGEr", 2, [MBI_Reg, MBI_Reg]>;
// def SETLEr :  TargInstr<"SETLEr", 2, [MBI_Reg, MBI_Reg]>;
// def SETGr  :  TargInstr<"SETGr", 2, [MBI_Reg, MBI_Reg]>;
def CMOVErr16 :  TargInstr<"CMOVE16rr", 2, [MBI_Reg, MBI_Reg]>;
def CMOVNErr32: TargInstr<"CMOVNE32rr", 2, [MBI_Reg, MBI_Reg]>;

def CMPrr8  :  TargInstr<"CMP8rr", 2, [MBI_Reg, MBI_Reg]>;
def CMPrr16 :  TargInstr<"CMP16rr", 2, [MBI_Reg, MBI_Reg]>;
def CMPrr32 :  TargInstr<"CMP32rr", 2, [MBI_Reg, MBI_Reg]>;
def CMPri8  :  TargInstr<"CMP8ri", 2, [MBI_Reg, MBI_ZImm]>;
def CMPri16 :  TargInstr<"CMP16ri", 2, [MBI_Reg, MBI_ZImm]>;
def CMPri32 :  TargInstr<"CMP32ri", 2, [MBI_Reg, MBI_ZImm]>;

def MOVSXr16r8 :  TargInstr<"MOVSX16rr8", 1, [MBI_Reg]>;
def MOVSXr32r8 :  TargInstr<"MOVSX32rr8", 1, [MBI_Reg]>;
def MOVSXr32r16: TargInstr<"MOVSX32rr16", 1, [MBI_Reg]>;
def MOVZXr16r8 :  TargInstr<"MOVZX16rr8", 1, [MBI_Reg]>;
def MOVZXr32r8 :  TargInstr<"MOVZX32rr8", 1, [MBI_Reg]>;
def MOVZXr32r16: TargInstr<"MOVZX32rr16", 1, [MBI_Reg]>;

def FP_REG_KILL : TargInstr<"FP_REG_KILL",0, []>;

def FpMOV  :  TargInstr<"FpMOV", 1, [MBI_Reg]>;
def FpADD  :  TargInstr<"FpADD", 2, [MBI_Reg, MBI_Reg]>;
def FpSUB  :  TargInstr<"FpSUB", 2, [MBI_Reg, MBI_Reg]>;
def FpMUL  :  TargInstr<"FpMUL", 2, [MBI_Reg, MBI_Reg]>;
def FpDIV  :  TargInstr<"FpDIV", 2, [MBI_Reg, MBI_Reg]>;
def FpUCOM :  TargInstr<"FpUCOM", 2, [MBI_Reg, MBI_Reg]>;
// def FpGETRESULT :  TargInstr<"FpGETRESULT", 2, [MBI_Reg, MBI_Reg]>;
def FpSETRESULT :  TargInstr<"FpSETRESULT", 1, [MBI_Reg]>;
// def FLDrr   :  TargInstr<"FLDrr", 2, [MBI_Reg, MBI_Reg]>;
def FLDr32  :  TargInstr<"FLD32r", 1, [MBI_ConstPoolRef]>; // MBI_FrameRef also? instructions can be passed different operands
def FLDr64  :  TargInstr<"FLD64r", 1, [MBI_ConstPoolRef]>;
// def FLDr80  :  TargInstr<"FLDr80", 2, [MBI_Reg, MBI_Reg]>;
def FILDr16 :  TargInstr<"FILD16r", 5, [MBI_FrameRef]>;
def FILDr32 :  TargInstr<"FILD32r", 5, [MBI_FrameRef]>;
def FILDr64 :  TargInstr<"FILD64r", 5, [MBI_FrameRef]>;
def FSTr32  :  TargInstr<"FST32r", 5, [MBI_FrameRef, MBI_Reg]>; // ?
// def FSTr64  :  TargInstr<"FST64r", 2, [MBI_Reg, MBI_Reg]>;
// def FSTPr32 :  TargInstr<"FSTP32r", 2, [MBI_Reg, MBI_Reg]>;
// def FSTPr64 :  TargInstr<"FSTP64r", 2, [MBI_Reg, MBI_Reg]>;
// def FSTPr80 :  TargInstr<"FSTP80r", 2, [MBI_Reg, MBI_Reg]>;
// def FSTrr   :  TargInstr<"FSTrr", 2, [MBI_Reg, MBI_Reg]>;
// def FSTPrr  :  TargInstr<"FSTPrr", 2, [MBI_Reg, MBI_Reg]>;
def FISTr16 :  TargInstr<"FIST16r", 5, [MBI_FrameRef, MBI_Reg]>;
def FISTr32 :  TargInstr<"FIST32r", 5, [MBI_FrameRef, MBI_Reg]>;
// def FISTPr16 :  TargInstr<"FISTP16r", 2, [MBI_Reg, MBI_Reg]>;
// def FISTPr32 :  TargInstr<"FISTP32r", 2, [MBI_Reg, MBI_Reg]>;
def FISTPr64 :  TargInstr<"FISTP64r", 5, [MBI_FrameRef, MBI_Reg]>;
// def FXCH :  TargInstr<"FXCH", 2, [MBI_Reg, MBI_Reg]>;
def FLD0 :  TargInstr<"FLD0", 0, []>;
def FLD1 :  TargInstr<"FLD1", 0, []>;
// def FADDST0r   :  TargInstr<"FADDST0r", 2, [MBI_Reg, MBI_Reg]>;
// def FADDrST0   :  TargInstr<"FADDrST0", 2, [MBI_Reg, MBI_Reg]>;
// def FADDPrST0  :  TargInstr<"FADDPrST0", 2, [MBI_Reg, MBI_Reg]>;
// def FSUBRST0r  :  TargInstr<"FSUBRST0r", 2, [MBI_Reg, MBI_Reg]>;
// def FSUBrST0   :  TargInstr<"FSUBrST0", 2, [MBI_Reg, MBI_Reg]>;
// def FSUBPrST0  :  TargInstr<"FSUBPrST0", 2, [MBI_Reg, MBI_Reg]>;
// def FSUBST0r   :  TargInstr<"FSUBST0r", 2, [MBI_Reg, MBI_Reg]>;
// def FSUBRrST0  :  TargInstr<"FSUBRrST0", 2, [MBI_Reg, MBI_Reg]>;
// def FSUBRPrST0 :  TargInstr<"FSUBRPrST0", 2, [MBI_Reg, MBI_Reg]>;
// def FMULST0r   :  TargInstr<"FMULST0r", 2, [MBI_Reg, MBI_Reg]>;
// def FMULrST0   :  TargInstr<"FMULrST0", 2, [MBI_Reg, MBI_Reg]>;
// def FMULPrST0  :  TargInstr<"FMULPrST0", 2, [MBI_Reg, MBI_Reg]>;
// def FDIVRST0r  :  TargInstr<"FDIVRST0r", 2, [MBI_Reg, MBI_Reg]>;
// def FDIVrST0   :  TargInstr<"FDIVrST0", 2, [MBI_Reg, MBI_Reg]>;
// def FDIVPrST0  :  TargInstr<"FDIVPrST0", 2, [MBI_Reg, MBI_Reg]>;
// def FDIVST0r   :  TargInstr<"FDIVST0r", 2, [MBI_Reg, MBI_Reg]>;
// def FDIVRrST0  :  TargInstr<"FDIVRrST0", 2, [MBI_Reg, MBI_Reg]>;
// def FDIVRPrST0 :  TargInstr<"FDIVRPrST0", 2, [MBI_Reg, MBI_Reg]>;
// def FUCOMr    :  TargInstr<"FUCOMr", 2, [MBI_Reg, MBI_Reg]>;
// def FUCOMPr   :  TargInstr<"FUCOMPr", 2, [MBI_Reg, MBI_Reg]>;
// def FUCOMPPr  :  TargInstr<"FUCOMPPr", 2, [MBI_Reg, MBI_Reg]>;
def FNSTSWr8  :  TargInstr<"FNSTSWr8", 0, []>; // store in "0" ?
// def FNSTCWm16 :  TargInstr<"FNSTCWm16", 2, [MBI_Reg, MBI_Reg]>;
def FLDCWm16  :  TargInstr<"FLDCWm16", 4, [MBI_FrameRef]>;

} //namespace
