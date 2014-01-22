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

//===----------------------------------------------------------------------===//
// R600
//===----------------------------------------------------------------------===//
def : Proc<"",           R600_VLIW5_Itin,
    [FeatureR600, FeatureVertexCache]>;

def : Proc<"r600",       R600_VLIW5_Itin,
    [FeatureR600 , FeatureVertexCache, FeatureWavefrontSize64]>;

def : Proc<"r630",       R600_VLIW5_Itin,
    [FeatureR600, FeatureVertexCache, FeatureWavefrontSize32]>;

def : Proc<"rs880",      R600_VLIW5_Itin,
    [FeatureR600, FeatureWavefrontSize16]>;

def : Proc<"rv670",      R600_VLIW5_Itin,
    [FeatureR600, FeatureFP64, FeatureVertexCache, FeatureWavefrontSize64]>;

//===----------------------------------------------------------------------===//
// R700
//===----------------------------------------------------------------------===//

def : Proc<"rv710",      R600_VLIW5_Itin,
    [FeatureR700, FeatureVertexCache, FeatureWavefrontSize32]>;

def : Proc<"rv730",      R600_VLIW5_Itin,
    [FeatureR700, FeatureVertexCache, FeatureWavefrontSize32]>;

def : Proc<"rv770",      R600_VLIW5_Itin,
    [FeatureR700, FeatureFP64, FeatureVertexCache, FeatureWavefrontSize64]>;

//===----------------------------------------------------------------------===//
// Evergreen
//===----------------------------------------------------------------------===//

def : Proc<"cedar",      R600_VLIW5_Itin,
    [FeatureEvergreen, FeatureVertexCache, FeatureWavefrontSize32]>;

def : Proc<"redwood",    R600_VLIW5_Itin,
    [FeatureEvergreen, FeatureVertexCache, FeatureWavefrontSize64]>;

def : Proc<"sumo",       R600_VLIW5_Itin,
    [FeatureEvergreen, FeatureWavefrontSize64]>;

def : Proc<"juniper",    R600_VLIW5_Itin,
    [FeatureEvergreen, FeatureVertexCache, FeatureWavefrontSize64]>;

def : Proc<"cypress",    R600_VLIW5_Itin,
    [FeatureEvergreen, FeatureFP64, FeatureVertexCache,
     FeatureWavefrontSize64]>;

//===----------------------------------------------------------------------===//
// Northern Islands
//===----------------------------------------------------------------------===//

def : Proc<"barts",      R600_VLIW5_Itin,
    [FeatureNorthernIslands, FeatureVertexCache]>;

def : Proc<"turks",      R600_VLIW5_Itin,
    [FeatureNorthernIslands, FeatureVertexCache]>;

def : Proc<"caicos",     R600_VLIW5_Itin,
    [FeatureNorthernIslands]>;

def : Proc<"cayman",     R600_VLIW4_Itin,
    [FeatureNorthernIslands, FeatureFP64, FeatureCaymanISA]>;

//===----------------------------------------------------------------------===//
// Southern Islands
//===----------------------------------------------------------------------===//

def : Proc<"SI",         SI_Itin, [FeatureSouthernIslands]>;

def : Proc<"tahiti",     SI_Itin, [FeatureSouthernIslands]>;

def : Proc<"pitcairn",   SI_Itin, [FeatureSouthernIslands]>;

def : Proc<"verde",      SI_Itin, [FeatureSouthernIslands]>;

def : Proc<"oland",      SI_Itin, [FeatureSouthernIslands]>;

def : Proc<"hainan",     SI_Itin, [FeatureSouthernIslands]>;

//===----------------------------------------------------------------------===//
// Sea Islands
//===----------------------------------------------------------------------===//

def : Proc<"bonaire",    SI_Itin, [FeatureSeaIslands]>;

def : Proc<"kabini",     SI_Itin, [FeatureSeaIslands]>;

def : Proc<"kaveri",     SI_Itin, [FeatureSeaIslands]>;

def : Proc<"hawaii",     SI_Itin, [FeatureSeaIslands]>;
