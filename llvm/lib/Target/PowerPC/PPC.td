//===-- PPC.td - Describe the PowerPC Target Machine -------*- tablegen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This is the top level entry point for the PowerPC target.
//
//===----------------------------------------------------------------------===//

// Get the target-independent interfaces which we are implementing.
//
include "llvm/Target/Target.td"

//===----------------------------------------------------------------------===//
// PowerPC Subtarget features.
//
 
//===----------------------------------------------------------------------===//
// CPU Directives                                                             //
//===----------------------------------------------------------------------===//

def Directive440 : SubtargetFeature<"", "DarwinDirective", "PPC::DIR_440", "">;
def Directive601 : SubtargetFeature<"", "DarwinDirective", "PPC::DIR_601", "">;
def Directive602 : SubtargetFeature<"", "DarwinDirective", "PPC::DIR_602", "">;
def Directive603 : SubtargetFeature<"", "DarwinDirective", "PPC::DIR_603", "">;
def Directive604 : SubtargetFeature<"", "DarwinDirective", "PPC::DIR_603", "">;
def Directive620 : SubtargetFeature<"", "DarwinDirective", "PPC::DIR_603", "">;
def Directive7400: SubtargetFeature<"", "DarwinDirective", "PPC::DIR_7400", "">;
def Directive750 : SubtargetFeature<"", "DarwinDirective", "PPC::DIR_750", "">;
def Directive970 : SubtargetFeature<"", "DarwinDirective", "PPC::DIR_970", "">;
def Directive32  : SubtargetFeature<"", "DarwinDirective", "PPC::DIR_32", "">;
def Directive64  : SubtargetFeature<"", "DarwinDirective", "PPC::DIR_64", "">;
def DirectiveA2  : SubtargetFeature<"", "DarwinDirective", "PPC::DIR_A2", "">;
def DirectiveE500mc : SubtargetFeature<"", "DarwinDirective",
                                       "PPC::DIR_E500mc", "">;
def DirectiveE5500  : SubtargetFeature<"", "DarwinDirective", 
                                       "PPC::DIR_E5500", "">;
def DirectivePwr3: SubtargetFeature<"", "DarwinDirective", "PPC::DIR_PWR3", "">;
def DirectivePwr4: SubtargetFeature<"", "DarwinDirective", "PPC::DIR_PWR4", "">;
def DirectivePwr5: SubtargetFeature<"", "DarwinDirective", "PPC::DIR_PWR5", "">;
def DirectivePwr5x: SubtargetFeature<"", "DarwinDirective", "PPC::DIR_PWR5X", "">;
def DirectivePwr6: SubtargetFeature<"", "DarwinDirective", "PPC::DIR_PWR6", "">;
def DirectivePwr6x: SubtargetFeature<"", "DarwinDirective", "PPC::DIR_PWR6X", "">;
def DirectivePwr7: SubtargetFeature<"", "DarwinDirective", "PPC::DIR_PWR7", "">;

def Feature64Bit     : SubtargetFeature<"64bit","Has64BitSupport", "true",
                                        "Enable 64-bit instructions">;
def Feature64BitRegs : SubtargetFeature<"64bitregs","Use64BitRegs", "true",
                              "Enable 64-bit registers usage for ppc32 [beta]">;
def FeatureAltivec   : SubtargetFeature<"altivec","HasAltivec", "true",
                                        "Enable Altivec instructions">;
def FeatureMFOCRF    : SubtargetFeature<"mfocrf","HasMFOCRF", "true",
                                        "Enable the MFOCRF instruction">;
def FeatureFSqrt     : SubtargetFeature<"fsqrt","HasFSQRT", "true",
                                        "Enable the fsqrt instruction">;
def FeatureFRE       : SubtargetFeature<"fre", "HasFRE", "true",
                                        "Enable the fre instruction">;
def FeatureFRES      : SubtargetFeature<"fres", "HasFRES", "true",
                                        "Enable the fres instruction">;
def FeatureFRSQRTE   : SubtargetFeature<"frsqrte", "HasFRSQRTE", "true",
                                        "Enable the frsqrte instruction">;
def FeatureFRSQRTES  : SubtargetFeature<"frsqrtes", "HasFRSQRTES", "true",
                                        "Enable the frsqrtes instruction">;
def FeatureRecipPrec : SubtargetFeature<"recipprec", "HasRecipPrec", "true",
                              "Assume higher precision reciprocal estimates">;
def FeatureSTFIWX    : SubtargetFeature<"stfiwx","HasSTFIWX", "true",
                                        "Enable the stfiwx instruction">;
def FeatureLFIWAX    : SubtargetFeature<"lfiwax","HasLFIWAX", "true",
                                        "Enable the lfiwax instruction">;
def FeatureFPRND     : SubtargetFeature<"fprnd", "HasFPRND", "true",
                                        "Enable the fri[mnpz] instructions">;
def FeatureFPCVT     : SubtargetFeature<"fpcvt", "HasFPCVT", "true",
  "Enable fc[ft]* (unsigned and single-precision) and lfiwzx instructions">;
def FeatureISEL      : SubtargetFeature<"isel","HasISEL", "true",
                                        "Enable the isel instruction">;
def FeaturePOPCNTD   : SubtargetFeature<"popcntd","HasPOPCNTD", "true",
                                        "Enable the popcnt[dw] instructions">;
def FeatureLDBRX     : SubtargetFeature<"ldbrx","HasLDBRX", "true",
                                        "Enable the ldbrx instruction">;
def FeatureBookE     : SubtargetFeature<"booke", "IsBookE", "true",
                                        "Enable Book E instructions">;
def FeatureQPX       : SubtargetFeature<"qpx","HasQPX", "true",
                                        "Enable QPX instructions">;

// Note: Future features to add when support is extended to more
// recent ISA levels:
//
// CMPB         p6, p6x, p7        cmpb
// DFP          p6, p6x, p7        decimal floating-point instructions
// POPCNTB      p5 through p7      popcntb and related instructions
// VSX          p7                 vector-scalar instruction set

//===----------------------------------------------------------------------===//
// Classes used for relation maps.
//===----------------------------------------------------------------------===//
// RecFormRel - Filter class used to relate non-record-form instructions with
// their record-form variants.
class RecFormRel;

//===----------------------------------------------------------------------===//
// Relation Map Definitions.
//===----------------------------------------------------------------------===//

def getRecordFormOpcode : InstrMapping {
  let FilterClass = "RecFormRel";
  // Instructions with the same BaseName and Interpretation64Bit values
  // form a row.
  let RowFields = ["BaseName", "Interpretation64Bit"];
  // Instructions with the same RC value form a column.
  let ColFields = ["RC"];
  // The key column are the non-record-form instructions.
  let KeyCol = ["0"];
  // Value columns RC=1
  let ValueCols = [["1"]];
}

def getNonRecordFormOpcode : InstrMapping {
  let FilterClass = "RecFormRel";
  // Instructions with the same BaseName and Interpretation64Bit values
  // form a row.
  let RowFields = ["BaseName", "Interpretation64Bit"];
  // Instructions with the same RC value form a column.
  let ColFields = ["RC"];
  // The key column are the record-form instructions.
  let KeyCol = ["1"];
  // Value columns are RC=0
  let ValueCols = [["0"]];
}

//===----------------------------------------------------------------------===//
// Register File Description
//===----------------------------------------------------------------------===//

include "PPCRegisterInfo.td"
include "PPCSchedule.td"
include "PPCInstrInfo.td"

//===----------------------------------------------------------------------===//
// PowerPC processors supported.
//

def : Processor<"generic", G3Itineraries, [Directive32]>;
def : Processor<"440", PPC440Itineraries, [Directive440, FeatureISEL,
                                           FeatureFRES, FeatureFRSQRTE,
                                           FeatureBookE]>;
def : Processor<"450", PPC440Itineraries, [Directive440, FeatureISEL,
                                           FeatureFRES, FeatureFRSQRTE,
                                           FeatureBookE]>;
def : Processor<"601", G3Itineraries, [Directive601]>;
def : Processor<"602", G3Itineraries, [Directive602]>;
def : Processor<"603", G3Itineraries, [Directive603,
                                       FeatureFRES, FeatureFRSQRTE]>;
def : Processor<"603e", G3Itineraries, [Directive603,
                                        FeatureFRES, FeatureFRSQRTE]>;
def : Processor<"603ev", G3Itineraries, [Directive603,
                                         FeatureFRES, FeatureFRSQRTE]>;
def : Processor<"604", G3Itineraries, [Directive604,
                                       FeatureFRES, FeatureFRSQRTE]>;
def : Processor<"604e", G3Itineraries, [Directive604,
                                        FeatureFRES, FeatureFRSQRTE]>;
def : Processor<"620", G3Itineraries, [Directive620,
                                       FeatureFRES, FeatureFRSQRTE]>;
def : Processor<"750", G4Itineraries, [Directive750,
                                       FeatureFRES, FeatureFRSQRTE]>;
def : Processor<"g3", G3Itineraries, [Directive750,
                                      FeatureFRES, FeatureFRSQRTE]>;
def : Processor<"7400", G4Itineraries, [Directive7400, FeatureAltivec,
                                        FeatureFRES, FeatureFRSQRTE]>;
def : Processor<"g4", G4Itineraries, [Directive7400, FeatureAltivec,
                                      FeatureFRES, FeatureFRSQRTE]>;
def : Processor<"7450", G4PlusItineraries, [Directive7400, FeatureAltivec,
                                            FeatureFRES, FeatureFRSQRTE]>;
def : Processor<"g4+", G4PlusItineraries, [Directive7400, FeatureAltivec,
                                           FeatureFRES, FeatureFRSQRTE]>;
def : ProcessorModel<"970", G5Model,
                  [Directive970, FeatureAltivec,
                   FeatureMFOCRF, FeatureFSqrt,
                   FeatureFRES, FeatureFRSQRTE, FeatureSTFIWX,
                   Feature64Bit /*, Feature64BitRegs */]>;
def : ProcessorModel<"g5", G5Model,
                  [Directive970, FeatureAltivec,
                   FeatureMFOCRF, FeatureFSqrt, FeatureSTFIWX,
                   FeatureFRES, FeatureFRSQRTE,
                   Feature64Bit /*, Feature64BitRegs */]>;
def : ProcessorModel<"e500mc", PPCE500mcModel,
                  [DirectiveE500mc, FeatureMFOCRF,
                   FeatureSTFIWX, FeatureBookE, FeatureISEL]>;
def : ProcessorModel<"e5500", PPCE5500Model,
                  [DirectiveE5500, FeatureMFOCRF, Feature64Bit,
                   FeatureSTFIWX, FeatureBookE, FeatureISEL]>;
def : ProcessorModel<"a2", PPCA2Model,
                  [DirectiveA2, FeatureBookE, FeatureMFOCRF,
                   FeatureFSqrt, FeatureFRE, FeatureFRES,
                   FeatureFRSQRTE, FeatureFRSQRTES, FeatureRecipPrec,
                   FeatureSTFIWX, FeatureLFIWAX,
                   FeatureFPRND, FeatureFPCVT, FeatureISEL,
                   FeaturePOPCNTD, FeatureLDBRX, Feature64Bit
               /*, Feature64BitRegs */]>;
def : ProcessorModel<"a2q", PPCA2Model,
                  [DirectiveA2, FeatureBookE, FeatureMFOCRF,
                   FeatureFSqrt, FeatureFRE, FeatureFRES,
                   FeatureFRSQRTE, FeatureFRSQRTES, FeatureRecipPrec,
                   FeatureSTFIWX, FeatureLFIWAX,
                   FeatureFPRND, FeatureFPCVT, FeatureISEL,
                   FeaturePOPCNTD, FeatureLDBRX, Feature64Bit
               /*, Feature64BitRegs */, FeatureQPX]>;
def : ProcessorModel<"pwr3", G5Model,
                  [DirectivePwr3, FeatureAltivec,
                   FeatureFRES, FeatureFRSQRTE, FeatureMFOCRF,
                   FeatureSTFIWX, Feature64Bit]>;
def : ProcessorModel<"pwr4", G5Model,
                  [DirectivePwr4, FeatureAltivec, FeatureMFOCRF,
                   FeatureFSqrt, FeatureFRES, FeatureFRSQRTE,
                   FeatureSTFIWX, Feature64Bit]>;
def : ProcessorModel<"pwr5", G5Model,
                  [DirectivePwr5, FeatureAltivec, FeatureMFOCRF,
                   FeatureFSqrt, FeatureFRE, FeatureFRES,
                   FeatureFRSQRTE, FeatureFRSQRTES,
                   FeatureSTFIWX, Feature64Bit]>;
def : ProcessorModel<"pwr5x", G5Model,
                  [DirectivePwr5x, FeatureAltivec, FeatureMFOCRF,
                   FeatureFSqrt, FeatureFRE, FeatureFRES,
                   FeatureFRSQRTE, FeatureFRSQRTES,
                   FeatureSTFIWX, FeatureFPRND, Feature64Bit]>;
def : ProcessorModel<"pwr6", G5Model,
                  [DirectivePwr6, FeatureAltivec,
                   FeatureMFOCRF, FeatureFSqrt, FeatureFRE,
                   FeatureFRES, FeatureFRSQRTE, FeatureFRSQRTES,
                   FeatureRecipPrec, FeatureSTFIWX, FeatureLFIWAX,
                   FeatureFPRND, Feature64Bit /*, Feature64BitRegs */]>;
def : ProcessorModel<"pwr6x", G5Model,
                  [DirectivePwr5x, FeatureAltivec, FeatureMFOCRF,
                   FeatureFSqrt, FeatureFRE, FeatureFRES,
                   FeatureFRSQRTE, FeatureFRSQRTES, FeatureRecipPrec,
                   FeatureSTFIWX, FeatureLFIWAX,
                   FeatureFPRND, Feature64Bit]>;
def : ProcessorModel<"pwr7", G5Model,
                  [DirectivePwr7, FeatureAltivec,
                   FeatureMFOCRF, FeatureFSqrt, FeatureFRE,
                   FeatureFRES, FeatureFRSQRTE, FeatureFRSQRTES,
                   FeatureRecipPrec, FeatureSTFIWX, FeatureLFIWAX,
                   FeatureFPRND, FeatureFPCVT, FeatureISEL,
                   FeaturePOPCNTD, FeatureLDBRX,
                   Feature64Bit /*, Feature64BitRegs */]>;
def : Processor<"ppc", G3Itineraries, [Directive32]>;
def : ProcessorModel<"ppc64", G5Model,
                  [Directive64, FeatureAltivec,
                   FeatureMFOCRF, FeatureFSqrt, FeatureFRES,
                   FeatureFRSQRTE, FeatureSTFIWX,
                   Feature64Bit /*, Feature64BitRegs */]>;

//===----------------------------------------------------------------------===//
// Calling Conventions
//===----------------------------------------------------------------------===//

include "PPCCallingConv.td"

def PPCInstrInfo : InstrInfo {
  let isLittleEndianEncoding = 1;
}

def PPCAsmWriter : AsmWriter {
  string AsmWriterClassName  = "InstPrinter";
  bit isMCAsmWriter = 1;
}

def PPCAsmParser : AsmParser {
  let ShouldEmitMatchRegisterName = 0;
}

def PPC : Target {
  // Information about the instructions.
  let InstructionSet = PPCInstrInfo;
  
  let AssemblyWriters = [PPCAsmWriter];
  let AssemblyParsers = [PPCAsmParser];
}
