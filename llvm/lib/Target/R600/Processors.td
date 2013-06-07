//===-- Processors.td - R600 Processor definitions ------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

class Proc<string Name, ProcessorItineraries itin, list<SubtargetFeature> Features>
: Processor<Name, itin, Features>;
def : Proc<"",           R600_VLIW5_Itin,
    [FeatureR600ALUInst, FeatureVertexCache, FeatureFetchLimit8]>;
def : Proc<"r600",       R600_VLIW5_Itin,
    [FeatureR600ALUInst , FeatureVertexCache, FeatureFetchLimit8]>;
def : Proc<"rs880",      R600_VLIW5_Itin,
    [FeatureR600ALUInst, FeatureFetchLimit8]>;
def : Proc<"rv670",      R600_VLIW5_Itin,
    [FeatureR600ALUInst, FeatureFP64, FeatureVertexCache, FeatureFetchLimit8]>;
def : Proc<"rv710",      R600_VLIW5_Itin,
    [FeatureVertexCache, FeatureFetchLimit16]>;
def : Proc<"rv730",      R600_VLIW5_Itin,
    [FeatureVertexCache, FeatureFetchLimit16]>;
def : Proc<"rv770",      R600_VLIW5_Itin,
    [FeatureFP64, FeatureVertexCache, FeatureFetchLimit16]>;
def : Proc<"cedar",      R600_VLIW5_Itin,
    [FeatureByteAddress, FeatureImages, FeatureVertexCache, FeatureFetchLimit16]>;
def : Proc<"redwood",    R600_VLIW5_Itin,
    [FeatureByteAddress, FeatureImages, FeatureVertexCache, FeatureFetchLimit16]>;
def : Proc<"sumo",       R600_VLIW5_Itin,
    [FeatureByteAddress, FeatureImages, FeatureFetchLimit16]>;
def : Proc<"juniper",    R600_VLIW5_Itin,
    [FeatureByteAddress, FeatureImages, FeatureVertexCache, FeatureFetchLimit16]>;
def : Proc<"cypress",    R600_VLIW5_Itin,
    [FeatureByteAddress, FeatureImages, FeatureFP64, FeatureVertexCache, FeatureFetchLimit16]>;
def : Proc<"barts",      R600_VLIW5_Itin,
    [FeatureByteAddress, FeatureImages, FeatureVertexCache, FeatureFetchLimit16]>;
def : Proc<"turks",      R600_VLIW5_Itin,
    [FeatureByteAddress, FeatureImages, FeatureVertexCache, FeatureFetchLimit16]>;
def : Proc<"caicos",     R600_VLIW5_Itin,
    [FeatureByteAddress, FeatureImages, FeatureFetchLimit16]>;
def : Proc<"cayman",     R600_VLIW4_Itin,
    [FeatureByteAddress, FeatureImages, FeatureFP64, FeatureFetchLimit16]>;

def : Proc<"SI",         SI_Itin, [Feature64BitPtr, FeatureFP64]>;
def : Proc<"tahiti",     SI_Itin, [Feature64BitPtr, FeatureFP64]>;
def : Proc<"pitcairn",   SI_Itin, [Feature64BitPtr, FeatureFP64]>;
def : Proc<"verde",      SI_Itin, [Feature64BitPtr, FeatureFP64]>;
def : Proc<"oland",      SI_Itin, [Feature64BitPtr, FeatureFP64]>;
def : Proc<"hainan",     SI_Itin, [Feature64BitPtr, FeatureFP64]>;
