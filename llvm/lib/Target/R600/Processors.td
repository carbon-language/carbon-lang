//===-- Processors.td - TODO: Add brief description -------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// AMDIL processors supported.
//
//===----------------------------------------------------------------------===//

class Proc<string Name, ProcessorItineraries itin, list<SubtargetFeature> Features>
: Processor<Name, itin, Features>;
def : Proc<"",           R600_EG_Itin,
    [FeatureR600ALUInst, FeatureVertexCache]>;
def : Proc<"r600",       R600_EG_Itin,
    [FeatureR600ALUInst , FeatureVertexCache]>;
def : Proc<"rs880",      R600_EG_Itin,
    [FeatureR600ALUInst]>;
def : Proc<"rv670",      R600_EG_Itin,
    [FeatureR600ALUInst, FeatureFP64, FeatureVertexCache]>;
def : Proc<"rv710",      R600_EG_Itin,
    [FeatureVertexCache]>;
def : Proc<"rv730",      R600_EG_Itin,
    [FeatureVertexCache]>;
def : Proc<"rv770",      R600_EG_Itin,
    [FeatureFP64, FeatureVertexCache]>;
def : Proc<"cedar",      R600_EG_Itin,
    [FeatureByteAddress, FeatureImages, FeatureVertexCache]>;
def : Proc<"redwood",    R600_EG_Itin,
    [FeatureByteAddress, FeatureImages, FeatureVertexCache]>;
def : Proc<"sumo",       R600_EG_Itin,
    [FeatureByteAddress, FeatureImages]>;
def : Proc<"juniper",    R600_EG_Itin,
    [FeatureByteAddress, FeatureImages, FeatureVertexCache]>;
def : Proc<"cypress",    R600_EG_Itin,
    [FeatureByteAddress, FeatureImages, FeatureFP64, FeatureVertexCache]>;
def : Proc<"barts",      R600_EG_Itin,
    [FeatureByteAddress, FeatureImages, FeatureVertexCache]>;
def : Proc<"turks",      R600_EG_Itin,
    [FeatureByteAddress, FeatureImages, FeatureVertexCache]>;
def : Proc<"caicos",     R600_EG_Itin,
    [FeatureByteAddress, FeatureImages]>;
def : Proc<"cayman",     R600_EG_Itin,
    [FeatureByteAddress, FeatureImages, FeatureFP64]>;def : Proc<"SI",         SI_Itin, [Feature64BitPtr, FeatureFP64]>;
def : Proc<"tahiti",     SI_Itin, [Feature64BitPtr, FeatureFP64]>;
def : Proc<"pitcairn",   SI_Itin, [Feature64BitPtr, FeatureFP64]>;
def : Proc<"verde",      SI_Itin, [Feature64BitPtr, FeatureFP64]>;
def : Proc<"oland",      SI_Itin, [Feature64BitPtr, FeatureFP64]>;
