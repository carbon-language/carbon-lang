//===- ARMSchedule.td - ARM Scheduling Definitions ---------*- tablegen -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Functional units across ARM processors
//
def FU_Issue   : FuncUnit; // issue
def FU_Pipe0   : FuncUnit; // pipeline 0
def FU_Pipe1   : FuncUnit; // pipeline 1
def FU_LdSt0   : FuncUnit; // pipeline 0 load/store
def FU_LdSt1   : FuncUnit; // pipeline 1 load/store
def FU_NPipe   : FuncUnit; // NEON ALU/MUL pipe
def FU_NLSPipe : FuncUnit; // NEON LS pipe

//===----------------------------------------------------------------------===//
// Instruction Itinerary classes used for ARM
//
def IIC_iALUx      : InstrItinClass;
def IIC_iALUi      : InstrItinClass;
def IIC_iALUr      : InstrItinClass;
def IIC_iALUsi     : InstrItinClass;
def IIC_iALUsr     : InstrItinClass;
def IIC_iUNAr      : InstrItinClass;
def IIC_iUNAsi     : InstrItinClass;
def IIC_iUNAsr     : InstrItinClass;
def IIC_iCMPi      : InstrItinClass;
def IIC_iCMPr      : InstrItinClass;
def IIC_iCMPsi     : InstrItinClass;
def IIC_iCMPsr     : InstrItinClass;
def IIC_iMOVi      : InstrItinClass;
def IIC_iMOVr      : InstrItinClass;
def IIC_iMOVsi     : InstrItinClass;
def IIC_iMOVsr     : InstrItinClass;
def IIC_iCMOVi     : InstrItinClass;
def IIC_iCMOVr     : InstrItinClass;
def IIC_iCMOVsi    : InstrItinClass;
def IIC_iCMOVsr    : InstrItinClass;
def IIC_iMUL16     : InstrItinClass;
def IIC_iMAC16     : InstrItinClass;
def IIC_iMUL32     : InstrItinClass;
def IIC_iMAC32     : InstrItinClass;
def IIC_iMUL64     : InstrItinClass;
def IIC_iMAC64     : InstrItinClass;
def IIC_iLoadi     : InstrItinClass;
def IIC_iLoadr     : InstrItinClass;
def IIC_iLoadsi    : InstrItinClass;
def IIC_iLoadiu    : InstrItinClass;
def IIC_iLoadru    : InstrItinClass;
def IIC_iLoadsiu   : InstrItinClass;
def IIC_iLoadm     : InstrItinClass;
def IIC_iStorei    : InstrItinClass;
def IIC_iStorer    : InstrItinClass;
def IIC_iStoresi   : InstrItinClass;
def IIC_iStoreiu   : InstrItinClass;
def IIC_iStoreru   : InstrItinClass;
def IIC_iStoresiu  : InstrItinClass;
def IIC_iStorem    : InstrItinClass;
def IIC_Br         : InstrItinClass;
def IIC_fpSTAT     : InstrItinClass;
def IIC_fpUNA32    : InstrItinClass;
def IIC_fpUNA64    : InstrItinClass;
def IIC_fpCMP32    : InstrItinClass;
def IIC_fpCMP64    : InstrItinClass;
def IIC_fpCVTSD    : InstrItinClass;
def IIC_fpCVTDS    : InstrItinClass;
def IIC_fpCVTIS    : InstrItinClass;
def IIC_fpCVTID    : InstrItinClass;
def IIC_fpCVTSI    : InstrItinClass;
def IIC_fpCVTDI    : InstrItinClass;
def IIC_fpALU32    : InstrItinClass;
def IIC_fpALU64    : InstrItinClass;
def IIC_fpMUL32    : InstrItinClass;
def IIC_fpMUL64    : InstrItinClass;
def IIC_fpMAC32    : InstrItinClass;
def IIC_fpMAC64    : InstrItinClass;
def IIC_fpDIV32    : InstrItinClass;
def IIC_fpDIV64    : InstrItinClass;
def IIC_fpSQRT32   : InstrItinClass;
def IIC_fpSQRT64   : InstrItinClass;
def IIC_fpLoad32   : InstrItinClass;
def IIC_fpLoad64   : InstrItinClass;
def IIC_fpLoadm    : InstrItinClass;
def IIC_fpStore32  : InstrItinClass;
def IIC_fpStore64  : InstrItinClass;
def IIC_fpStorem   : InstrItinClass;
def IIC_VLD1       : InstrItinClass;
def IIC_VLD2       : InstrItinClass;
def IIC_VLD3       : InstrItinClass;
def IIC_VLD4       : InstrItinClass;
def IIC_VST        : InstrItinClass;
def IIC_VUNAD      : InstrItinClass;
def IIC_VUNAQ      : InstrItinClass;
def IIC_VBIND      : InstrItinClass;
def IIC_VBINQ      : InstrItinClass;
def IIC_VMOVImm    : InstrItinClass;
def IIC_VMOVD      : InstrItinClass;
def IIC_VMOVQ      : InstrItinClass;
def IIC_VMOVIS     : InstrItinClass;
def IIC_VMOVID     : InstrItinClass;
def IIC_VMOVISL    : InstrItinClass;
def IIC_VMOVSI     : InstrItinClass;
def IIC_VMOVDI     : InstrItinClass;
def IIC_VPERMD     : InstrItinClass;
def IIC_VPERMQ     : InstrItinClass;
def IIC_VPERMQ3    : InstrItinClass;
def IIC_VMACD      : InstrItinClass;
def IIC_VMACQ      : InstrItinClass;
def IIC_VRECSD     : InstrItinClass;
def IIC_VRECSQ     : InstrItinClass;
def IIC_VCNTiD     : InstrItinClass;
def IIC_VCNTiQ     : InstrItinClass;
def IIC_VUNAiD     : InstrItinClass;
def IIC_VUNAiQ     : InstrItinClass;
def IIC_VQUNAiD    : InstrItinClass;
def IIC_VQUNAiQ    : InstrItinClass;
def IIC_VBINiD     : InstrItinClass;
def IIC_VBINiQ     : InstrItinClass;
def IIC_VSUBiD     : InstrItinClass;
def IIC_VSUBiQ     : InstrItinClass;
def IIC_VBINi4D    : InstrItinClass;
def IIC_VBINi4Q    : InstrItinClass;
def IIC_VSHLiD     : InstrItinClass;
def IIC_VSHLiQ     : InstrItinClass;
def IIC_VSHLi4D    : InstrItinClass;
def IIC_VSHLi4Q    : InstrItinClass;
def IIC_VPALiD     : InstrItinClass;
def IIC_VPALiQ     : InstrItinClass;
def IIC_VMULi16D   : InstrItinClass;
def IIC_VMULi32D   : InstrItinClass;
def IIC_VMULi16Q   : InstrItinClass;
def IIC_VMULi32Q   : InstrItinClass;
def IIC_VMACi16D   : InstrItinClass;
def IIC_VMACi32D   : InstrItinClass;
def IIC_VMACi16Q   : InstrItinClass;
def IIC_VMACi32Q   : InstrItinClass;
def IIC_VEXTD      : InstrItinClass;
def IIC_VEXTQ      : InstrItinClass;
def IIC_VTB1       : InstrItinClass;
def IIC_VTB2       : InstrItinClass;
def IIC_VTB3       : InstrItinClass;
def IIC_VTB4       : InstrItinClass;
def IIC_VTBX1      : InstrItinClass;
def IIC_VTBX2      : InstrItinClass;
def IIC_VTBX3      : InstrItinClass;
def IIC_VTBX4      : InstrItinClass;

//===----------------------------------------------------------------------===//
// Processor instruction itineraries.

def GenericItineraries : ProcessorItineraries<[]>;


include "ARMScheduleV6.td"
include "ARMScheduleV7.td"
