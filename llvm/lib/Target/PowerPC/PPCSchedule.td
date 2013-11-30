//===-- PPCSchedule.td - PowerPC Scheduling Definitions ----*- tablegen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Instruction Itinerary classes used for PowerPC
//
def IIC_IntSimple    : InstrItinClass;
def IIC_IntGeneral   : InstrItinClass;
def IIC_IntCompare   : InstrItinClass;
def IIC_IntDivD      : InstrItinClass;
def IIC_IntDivW      : InstrItinClass;
def IIC_IntMFFS      : InstrItinClass;
def IIC_IntMFVSCR    : InstrItinClass;
def IIC_IntMTFSB0    : InstrItinClass;
def IIC_IntMTSRD     : InstrItinClass;
def IIC_IntMulHD     : InstrItinClass;
def IIC_IntMulHW     : InstrItinClass;
def IIC_IntMulHWU    : InstrItinClass;
def IIC_IntMulLI     : InstrItinClass;
def IIC_IntRFID      : InstrItinClass;
def IIC_IntRotateD   : InstrItinClass;
def IIC_IntRotateDI  : InstrItinClass;
def IIC_IntRotate    : InstrItinClass;
def IIC_IntShift     : InstrItinClass;
def IIC_IntTrapD     : InstrItinClass;
def IIC_IntTrapW     : InstrItinClass;
def IIC_BrB          : InstrItinClass;
def IIC_BrCR         : InstrItinClass;
def IIC_BrMCR        : InstrItinClass;
def IIC_BrMCRX       : InstrItinClass;
def IIC_LdStDCBA     : InstrItinClass;
def IIC_LdStDCBF     : InstrItinClass;
def IIC_LdStDCBI     : InstrItinClass;
def IIC_LdStLoad     : InstrItinClass;
def IIC_LdStLoadUpd  : InstrItinClass;
def IIC_LdStLoadUpdX : InstrItinClass;
def IIC_LdStStore    : InstrItinClass;
def IIC_LdStStoreUpd : InstrItinClass;
def IIC_LdStDSS      : InstrItinClass;
def IIC_LdStICBI     : InstrItinClass;
def IIC_LdStLD       : InstrItinClass;
def IIC_LdStLDU      : InstrItinClass;
def IIC_LdStLDUX     : InstrItinClass;
def IIC_LdStLDARX    : InstrItinClass;
def IIC_LdStLFD      : InstrItinClass;
def IIC_LdStLFDU     : InstrItinClass;
def IIC_LdStLFDUX    : InstrItinClass;
def IIC_LdStLHA      : InstrItinClass;
def IIC_LdStLHAU     : InstrItinClass;
def IIC_LdStLHAUX    : InstrItinClass;
def IIC_LdStLMW      : InstrItinClass;
def IIC_LdStLVecX    : InstrItinClass;
def IIC_LdStLWA      : InstrItinClass;
def IIC_LdStLWARX    : InstrItinClass;
def IIC_LdStSLBIA    : InstrItinClass;
def IIC_LdStSLBIE    : InstrItinClass;
def IIC_LdStSTD      : InstrItinClass;
def IIC_LdStSTDCX    : InstrItinClass;
def IIC_LdStSTDU     : InstrItinClass;
def IIC_LdStSTDUX    : InstrItinClass;
def IIC_LdStSTFD     : InstrItinClass;
def IIC_LdStSTFDU    : InstrItinClass;
def IIC_LdStSTVEBX   : InstrItinClass;
def IIC_LdStSTWCX    : InstrItinClass;
def IIC_LdStSync     : InstrItinClass;
def IIC_SprISYNC     : InstrItinClass;
def IIC_SprMFSR      : InstrItinClass;
def IIC_SprMTMSR     : InstrItinClass;
def IIC_SprMTSR      : InstrItinClass;
def IIC_SprTLBSYNC   : InstrItinClass;
def IIC_SprMFCR      : InstrItinClass;
def IIC_SprMFCRF     : InstrItinClass;
def IIC_SprMFMSR     : InstrItinClass;
def IIC_SprMFSPR     : InstrItinClass;
def IIC_SprMFTB      : InstrItinClass;
def IIC_SprMTSPR     : InstrItinClass;
def IIC_SprMTSRIN    : InstrItinClass;
def IIC_SprRFI       : InstrItinClass;
def IIC_SprSC        : InstrItinClass;
def IIC_FPGeneral    : InstrItinClass;
def IIC_FPAddSub     : InstrItinClass;
def IIC_FPCompare    : InstrItinClass;
def IIC_FPDivD       : InstrItinClass;
def IIC_FPDivS       : InstrItinClass;
def IIC_FPFused      : InstrItinClass;
def IIC_FPRes        : InstrItinClass;
def IIC_FPSqrtD      : InstrItinClass;
def IIC_FPSqrtS      : InstrItinClass;
def IIC_VecGeneral   : InstrItinClass;
def IIC_VecFP        : InstrItinClass;
def IIC_VecFPCompare : InstrItinClass;
def IIC_VecComplex   : InstrItinClass;
def IIC_VecPerm      : InstrItinClass;
def IIC_VecFPRound   : InstrItinClass;
def IIC_VecVSL       : InstrItinClass;
def IIC_VecVSR       : InstrItinClass;
def IIC_SprMTMSRD    : InstrItinClass;
def IIC_SprSLIE      : InstrItinClass;
def IIC_SprSLBIE     : InstrItinClass;
def IIC_SprSLBMTE    : InstrItinClass;
def IIC_SprSLBMFEE   : InstrItinClass;
def IIC_SprSLBIA     : InstrItinClass;
def IIC_SprTLBIEL    : InstrItinClass;
def IIC_SprTLBIE     : InstrItinClass;

//===----------------------------------------------------------------------===//
// Processor instruction itineraries.

include "PPCScheduleG3.td"
include "PPCSchedule440.td"
include "PPCScheduleG4.td"
include "PPCScheduleG4Plus.td"
include "PPCScheduleG5.td"
include "PPCScheduleP7.td"
include "PPCScheduleA2.td"
include "PPCScheduleE500mc.td"
include "PPCScheduleE5500.td"

//===----------------------------------------------------------------------===//
// Instruction to itinerary class map - When add new opcodes to the supported
// set, refer to the following table to determine which itinerary class the
// opcode belongs.
//
//    opcode     itinerary class
//    ======     ===============
//    add        IIC_IntSimple
//    addc       IIC_IntGeneral
//    adde       IIC_IntGeneral
//    addi       IIC_IntSimple
//    addic      IIC_IntGeneral
//    addic.     IIC_IntGeneral
//    addis      IIC_IntSimple
//    addme      IIC_IntGeneral
//    addze      IIC_IntGeneral
//    and        IIC_IntSimple
//    andc       IIC_IntSimple
//    andi.      IIC_IntGeneral
//    andis.     IIC_IntGeneral
//    b          IIC_BrB
//    bc         IIC_BrB
//    bcctr      IIC_BrB
//    bclr       IIC_BrB
//    cmp        IIC_IntCompare
//    cmpi       IIC_IntCompare
//    cmpl       IIC_IntCompare
//    cmpli      IIC_IntCompare
//    cntlzd     IIC_IntRotateD
//    cntlzw     IIC_IntGeneral
//    crand      IIC_BrCR
//    crandc     IIC_BrCR
//    creqv      IIC_BrCR
//    crnand     IIC_BrCR
//    crnor      IIC_BrCR
//    cror       IIC_BrCR
//    crorc      IIC_BrCR
//    crxor      IIC_BrCR
//    dcba       IIC_LdStDCBA
//    dcbf       IIC_LdStDCBF
//    dcbi       IIC_LdStDCBI
//    dcbst      IIC_LdStDCBF
//    dcbt       IIC_LdStLoad
//    dcbtst     IIC_LdStLoad
//    dcbz       IIC_LdStDCBF
//    divd       IIC_IntDivD
//    divdu      IIC_IntDivD
//    divw       IIC_IntDivW
//    divwu      IIC_IntDivW
//    dss        IIC_LdStDSS
//    dst        IIC_LdStDSS
//    dstst      IIC_LdStDSS
//    eciwx      IIC_LdStLoad
//    ecowx      IIC_LdStLoad
//    eieio      IIC_LdStLoad
//    eqv        IIC_IntSimple
//    extsb      IIC_IntSimple
//    extsh      IIC_IntSimple
//    extsw      IIC_IntSimple
//    fabs       IIC_FPGeneral
//    fadd       IIC_FPAddSub
//    fadds      IIC_FPGeneral
//    fcfid      IIC_FPGeneral
//    fcmpo      IIC_FPCompare
//    fcmpu      IIC_FPCompare
//    fctid      IIC_FPGeneral
//    fctidz     IIC_FPGeneral
//    fctiw      IIC_FPGeneral
//    fctiwz     IIC_FPGeneral
//    fdiv       IIC_FPDivD
//    fdivs      IIC_FPDivS
//    fmadd      IIC_FPFused
//    fmadds     IIC_FPGeneral
//    fmr        IIC_FPGeneral
//    fmsub      IIC_FPFused
//    fmsubs     IIC_FPGeneral
//    fmul       IIC_FPFused
//    fmuls      IIC_FPGeneral
//    fnabs      IIC_FPGeneral
//    fneg       IIC_FPGeneral
//    fnmadd     IIC_FPFused
//    fnmadds    IIC_FPGeneral
//    fnmsub     IIC_FPFused
//    fnmsubs    IIC_FPGeneral
//    fres       IIC_FPRes
//    frsp       IIC_FPGeneral
//    frsqrte    IIC_FPGeneral
//    fsel       IIC_FPGeneral
//    fsqrt      IIC_FPSqrtD
//    fsqrts     IIC_FPSqrtS
//    fsub       IIC_FPAddSub
//    fsubs      IIC_FPGeneral
//    icbi       IIC_LdStICBI
//    isync      IIC_SprISYNC
//    lbz        IIC_LdStLoad
//    lbzu       IIC_LdStLoadUpd
//    lbzux      IIC_LdStLoadUpdX
//    lbzx       IIC_LdStLoad
//    ld         IIC_LdStLD
//    ldarx      IIC_LdStLDARX
//    ldu        IIC_LdStLDU
//    ldux       IIC_LdStLDUX
//    ldx        IIC_LdStLD
//    lfd        IIC_LdStLFD
//    lfdu       IIC_LdStLFDU
//    lfdux      IIC_LdStLFDUX
//    lfdx       IIC_LdStLFD
//    lfs        IIC_LdStLFD
//    lfsu       IIC_LdStLFDU
//    lfsux      IIC_LdStLFDUX
//    lfsx       IIC_LdStLFD
//    lha        IIC_LdStLHA
//    lhau       IIC_LdStLHAU
//    lhaux      IIC_LdStLHAUX
//    lhax       IIC_LdStLHA
//    lhbrx      IIC_LdStLoad
//    lhz        IIC_LdStLoad
//    lhzu       IIC_LdStLoadUpd
//    lhzux      IIC_LdStLoadUpdX
//    lhzx       IIC_LdStLoad
//    lmw        IIC_LdStLMW
//    lswi       IIC_LdStLMW
//    lswx       IIC_LdStLMW
//    lvebx      IIC_LdStLVecX
//    lvehx      IIC_LdStLVecX
//    lvewx      IIC_LdStLVecX
//    lvsl       IIC_LdStLVecX
//    lvsr       IIC_LdStLVecX
//    lvx        IIC_LdStLVecX
//    lvxl       IIC_LdStLVecX
//    lwa        IIC_LdStLWA
//    lwarx      IIC_LdStLWARX
//    lwaux      IIC_LdStLHAUX
//    lwax       IIC_LdStLHA
//    lwbrx      IIC_LdStLoad
//    lwz        IIC_LdStLoad
//    lwzu       IIC_LdStLoadUpd
//    lwzux      IIC_LdStLoadUpdX
//    lwzx       IIC_LdStLoad
//    mcrf       IIC_BrMCR
//    mcrfs      IIC_FPGeneral
//    mcrxr      IIC_BrMCRX
//    mfcr       IIC_SprMFCR
//    mffs       IIC_IntMFFS
//    mfmsr      IIC_SprMFMSR
//    mfspr      IIC_SprMFSPR
//    mfsr       IIC_SprMFSR
//    mfsrin     IIC_SprMFSR
//    mftb       IIC_SprMFTB
//    mfvscr     IIC_IntMFVSCR
//    mtcrf      IIC_BrMCRX
//    mtfsb0     IIC_IntMTFSB0
//    mtfsb1     IIC_IntMTFSB0
//    mtfsf      IIC_IntMTFSB0
//    mtfsfi     IIC_IntMTFSB0
//    mtmsr      IIC_SprMTMSR
//    mtmsrd     IIC_LdStLD
//    mtspr      IIC_SprMTSPR
//    mtsr       IIC_SprMTSR
//    mtsrd      IIC_IntMTSRD
//    mtsrdin    IIC_IntMTSRD
//    mtsrin     IIC_SprMTSRIN
//    mtvscr     IIC_IntMFVSCR
//    mulhd      IIC_IntMulHD
//    mulhdu     IIC_IntMulHD
//    mulhw      IIC_IntMulHW
//    mulhwu     IIC_IntMulHWU
//    mulld      IIC_IntMulHD
//    mulli      IIC_IntMulLI
//    mullw      IIC_IntMulHW
//    nand       IIC_IntSimple
//    neg        IIC_IntSimple
//    nor        IIC_IntSimple
//    or         IIC_IntSimple
//    orc        IIC_IntSimple
//    ori        IIC_IntSimple
//    oris       IIC_IntSimple
//    rfi        IIC_SprRFI
//    rfid       IIC_IntRFID
//    rldcl      IIC_IntRotateD
//    rldcr      IIC_IntRotateD
//    rldic      IIC_IntRotateDI
//    rldicl     IIC_IntRotateDI
//    rldicr     IIC_IntRotateDI
//    rldimi     IIC_IntRotateDI
//    rlwimi     IIC_IntRotate
//    rlwinm     IIC_IntGeneral
//    rlwnm      IIC_IntGeneral
//    sc         IIC_SprSC
//    slbia      IIC_LdStSLBIA
//    slbie      IIC_LdStSLBIE
//    sld        IIC_IntRotateD
//    slw        IIC_IntGeneral
//    srad       IIC_IntRotateD
//    sradi      IIC_IntRotateDI
//    sraw       IIC_IntShift
//    srawi      IIC_IntShift
//    srd        IIC_IntRotateD
//    srw        IIC_IntGeneral
//    stb        IIC_LdStStore
//    stbu       IIC_LdStStoreUpd
//    stbux      IIC_LdStStoreUpd
//    stbx       IIC_LdStStore
//    std        IIC_LdStSTD
//    stdcx.     IIC_LdStSTDCX
//    stdu       IIC_LdStSTDU
//    stdux      IIC_LdStSTDUX
//    stdx       IIC_LdStSTD
//    stfd       IIC_LdStSTFD
//    stfdu      IIC_LdStSTFDU
//    stfdux     IIC_LdStSTFDU
//    stfdx      IIC_LdStSTFD
//    stfiwx     IIC_LdStSTFD
//    stfs       IIC_LdStSTFD
//    stfsu      IIC_LdStSTFDU
//    stfsux     IIC_LdStSTFDU
//    stfsx      IIC_LdStSTFD
//    sth        IIC_LdStStore
//    sthbrx     IIC_LdStStore
//    sthu       IIC_LdStStoreUpd
//    sthux      IIC_LdStStoreUpd
//    sthx       IIC_LdStStore
//    stmw       IIC_LdStLMW
//    stswi      IIC_LdStLMW
//    stswx      IIC_LdStLMW
//    stvebx     IIC_LdStSTVEBX
//    stvehx     IIC_LdStSTVEBX
//    stvewx     IIC_LdStSTVEBX
//    stvx       IIC_LdStSTVEBX
//    stvxl      IIC_LdStSTVEBX
//    stw        IIC_LdStStore
//    stwbrx     IIC_LdStStore
//    stwcx.     IIC_LdStSTWCX
//    stwu       IIC_LdStStoreUpd
//    stwux      IIC_LdStStoreUpd
//    stwx       IIC_LdStStore
//    subf       IIC_IntGeneral
//    subfc      IIC_IntGeneral
//    subfe      IIC_IntGeneral
//    subfic     IIC_IntGeneral
//    subfme     IIC_IntGeneral
//    subfze     IIC_IntGeneral
//    sync       IIC_LdStSync
//    td         IIC_IntTrapD
//    tdi        IIC_IntTrapD
//    tlbia      IIC_LdStSLBIA
//    tlbie      IIC_LdStDCBF
//    tlbsync    IIC_SprTLBSYNC
//    tw         IIC_IntTrapW
//    twi        IIC_IntTrapW
//    vaddcuw    IIC_VecGeneral
//    vaddfp     IIC_VecFP
//    vaddsbs    IIC_VecGeneral
//    vaddshs    IIC_VecGeneral
//    vaddsws    IIC_VecGeneral
//    vaddubm    IIC_VecGeneral
//    vaddubs    IIC_VecGeneral
//    vadduhm    IIC_VecGeneral
//    vadduhs    IIC_VecGeneral
//    vadduwm    IIC_VecGeneral
//    vadduws    IIC_VecGeneral
//    vand       IIC_VecGeneral
//    vandc      IIC_VecGeneral
//    vavgsb     IIC_VecGeneral
//    vavgsh     IIC_VecGeneral
//    vavgsw     IIC_VecGeneral
//    vavgub     IIC_VecGeneral
//    vavguh     IIC_VecGeneral
//    vavguw     IIC_VecGeneral
//    vcfsx      IIC_VecFP
//    vcfux      IIC_VecFP
//    vcmpbfp    IIC_VecFPCompare
//    vcmpeqfp   IIC_VecFPCompare
//    vcmpequb   IIC_VecGeneral
//    vcmpequh   IIC_VecGeneral
//    vcmpequw   IIC_VecGeneral
//    vcmpgefp   IIC_VecFPCompare
//    vcmpgtfp   IIC_VecFPCompare
//    vcmpgtsb   IIC_VecGeneral
//    vcmpgtsh   IIC_VecGeneral
//    vcmpgtsw   IIC_VecGeneral
//    vcmpgtub   IIC_VecGeneral
//    vcmpgtuh   IIC_VecGeneral
//    vcmpgtuw   IIC_VecGeneral
//    vctsxs     IIC_VecFP
//    vctuxs     IIC_VecFP
//    vexptefp   IIC_VecFP
//    vlogefp    IIC_VecFP
//    vmaddfp    IIC_VecFP
//    vmaxfp     IIC_VecFPCompare
//    vmaxsb     IIC_VecGeneral
//    vmaxsh     IIC_VecGeneral
//    vmaxsw     IIC_VecGeneral
//    vmaxub     IIC_VecGeneral
//    vmaxuh     IIC_VecGeneral
//    vmaxuw     IIC_VecGeneral
//    vmhaddshs  IIC_VecComplex
//    vmhraddshs IIC_VecComplex
//    vminfp     IIC_VecFPCompare
//    vminsb     IIC_VecGeneral
//    vminsh     IIC_VecGeneral
//    vminsw     IIC_VecGeneral
//    vminub     IIC_VecGeneral
//    vminuh     IIC_VecGeneral
//    vminuw     IIC_VecGeneral
//    vmladduhm  IIC_VecComplex
//    vmrghb     IIC_VecPerm
//    vmrghh     IIC_VecPerm
//    vmrghw     IIC_VecPerm
//    vmrglb     IIC_VecPerm
//    vmrglh     IIC_VecPerm
//    vmrglw     IIC_VecPerm
//    vmsubfp    IIC_VecFP
//    vmsummbm   IIC_VecComplex
//    vmsumshm   IIC_VecComplex
//    vmsumshs   IIC_VecComplex
//    vmsumubm   IIC_VecComplex
//    vmsumuhm   IIC_VecComplex
//    vmsumuhs   IIC_VecComplex
//    vmulesb    IIC_VecComplex
//    vmulesh    IIC_VecComplex
//    vmuleub    IIC_VecComplex
//    vmuleuh    IIC_VecComplex
//    vmulosb    IIC_VecComplex
//    vmulosh    IIC_VecComplex
//    vmuloub    IIC_VecComplex
//    vmulouh    IIC_VecComplex
//    vnor       IIC_VecGeneral
//    vor        IIC_VecGeneral
//    vperm      IIC_VecPerm
//    vpkpx      IIC_VecPerm
//    vpkshss    IIC_VecPerm
//    vpkshus    IIC_VecPerm
//    vpkswss    IIC_VecPerm
//    vpkswus    IIC_VecPerm
//    vpkuhum    IIC_VecPerm
//    vpkuhus    IIC_VecPerm
//    vpkuwum    IIC_VecPerm
//    vpkuwus    IIC_VecPerm
//    vrefp      IIC_VecFPRound
//    vrfim      IIC_VecFPRound
//    vrfin      IIC_VecFPRound
//    vrfip      IIC_VecFPRound
//    vrfiz      IIC_VecFPRound
//    vrlb       IIC_VecGeneral
//    vrlh       IIC_VecGeneral
//    vrlw       IIC_VecGeneral
//    vrsqrtefp  IIC_VecFP
//    vsel       IIC_VecGeneral
//    vsl        IIC_VecVSL
//    vslb       IIC_VecGeneral
//    vsldoi     IIC_VecPerm
//    vslh       IIC_VecGeneral
//    vslo       IIC_VecPerm
//    vslw       IIC_VecGeneral
//    vspltb     IIC_VecPerm
//    vsplth     IIC_VecPerm
//    vspltisb   IIC_VecPerm
//    vspltish   IIC_VecPerm
//    vspltisw   IIC_VecPerm
//    vspltw     IIC_VecPerm
//    vsr        IIC_VecVSR
//    vsrab      IIC_VecGeneral
//    vsrah      IIC_VecGeneral
//    vsraw      IIC_VecGeneral
//    vsrb       IIC_VecGeneral
//    vsrh       IIC_VecGeneral
//    vsro       IIC_VecPerm
//    vsrw       IIC_VecGeneral
//    vsubcuw    IIC_VecGeneral
//    vsubfp     IIC_VecFP
//    vsubsbs    IIC_VecGeneral
//    vsubshs    IIC_VecGeneral
//    vsubsws    IIC_VecGeneral
//    vsububm    IIC_VecGeneral
//    vsububs    IIC_VecGeneral
//    vsubuhm    IIC_VecGeneral
//    vsubuhs    IIC_VecGeneral
//    vsubuwm    IIC_VecGeneral
//    vsubuws    IIC_VecGeneral
//    vsum2sws   IIC_VecComplex
//    vsum4sbs   IIC_VecComplex
//    vsum4shs   IIC_VecComplex
//    vsum4ubs   IIC_VecComplex
//    vsumsws    IIC_VecComplex
//    vupkhpx    IIC_VecPerm
//    vupkhsb    IIC_VecPerm
//    vupkhsh    IIC_VecPerm
//    vupklpx    IIC_VecPerm
//    vupklsb    IIC_VecPerm
//    vupklsh    IIC_VecPerm
//    vxor       IIC_VecGeneral
//    xor        IIC_IntSimple
//    xori       IIC_IntSimple
//    xoris      IIC_IntSimple
//
