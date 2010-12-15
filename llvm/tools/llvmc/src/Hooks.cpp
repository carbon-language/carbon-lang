#include "llvm/ADT/StringMap.h"

#include <string>
#include <vector>

namespace hooks {

/// NUM_KEYS - Calculate the size of a const char* array.
#define NUM_KEYS(Keys) sizeof(Keys) / sizeof(const char*)

// See http://graphics.stanford.edu/~seander/bithacks.html#RoundUpPowerOf2
inline unsigned NextHighestPowerOf2 (unsigned i) {
  --i;
  i |= i >> 1;
  i |= i >> 2;
  i |= i >> 4;
  i |= i >> 8;
  i |= i >> 16;
  ++i;
  return i;
}

typedef std::vector<std::string> StrVec;
typedef llvm::StringMap<const char*> ArgMap;

/// AddPlusOrMinus - Convert 'no-foo' to '-foo' and 'foo' to '+foo'.
void AddPlusOrMinus (const std::string& Arg, std::string& out) {
  if (Arg.find("no-") == 0 && Arg[3] != 0) {
    out += '-';
    out += Arg.c_str() + 3;
  }
  else {
    out += '+';
    out += Arg;
  }
}

// -march values that need to be special-cased.
const char* MArchKeysARM[] = { "armv4t", "armv5t", "armv5te", "armv6",
                               "armv6-m", "armv6t2", "armv7-a", "armv7-m" };
const char* MArchValuesARM[] = { "v4t", "v5t", "v5te", "v6", "v6m", "v6t2",
                                 "v7a", "v7m" };
const unsigned MArchNumKeysARM = NUM_KEYS(MArchKeysARM);
const unsigned MArchMapSize = NextHighestPowerOf2(MArchNumKeysARM);

// -march values that should be forwarded as -mcpu
const char* MArchMCpuKeysARM[] = { "iwmmxt", "ep9312" };
const char* MArchMCpuValuesARM[] = { "iwmmxt", "ep9312"};
const unsigned MArchMCpuNumKeysARM = NUM_KEYS(MArchMCpuKeysARM);
const unsigned MArchMCpuMapSize = NextHighestPowerOf2(MArchMCpuNumKeysARM);


void FillInArgMap(ArgMap& Args, const char* Keys[],
                  const char* Values[], unsigned NumKeys)
{
  for (unsigned i = 0; i < NumKeys; ++i) {
    // Explicit cast to StringRef here is necessary to pick up the right
    // overload.
    Args.GetOrCreateValue(llvm::StringRef(Keys[i]), Values[i]);
  }
}

/// ConvertMArchToMAttr - Convert -march from the gcc dialect to
/// something llc can understand.
std::string ConvertMArchToMAttr(const StrVec& Opts) {
  static ArgMap MArchMap(MArchMapSize);
  static ArgMap MArchMCpuMap(MArchMapSize);
  static bool StaticDataInitialized = false;

  if (!StaticDataInitialized) {
    FillInArgMap(MArchMap, MArchKeysARM, MArchValuesARM, MArchNumKeysARM);
    FillInArgMap(MArchMCpuMap, MArchMCpuKeysARM,
                 MArchMCpuValuesARM, MArchMCpuNumKeysARM);
    StaticDataInitialized = true;
  }

  std::string mattr("-mattr=");
  std::string mcpu("-mcpu=");
  bool mattrTouched = false;
  bool mcpuTouched = false;

  for (StrVec::const_iterator B = Opts.begin(), E = Opts.end(); B!=E; ++B) {
    const std::string& Arg = *B;

    // Check if the argument should be forwarded to -mcpu instead of -mattr.
    {
      ArgMap::const_iterator I = MArchMCpuMap.find(Arg);

      if (I != MArchMCpuMap.end()) {
        mcpuTouched = true;
        mcpu += I->getValue();
        continue;
      }
    }

    if (mattrTouched)
      mattr += ",";

    // Check if the argument is a special case.
    {
      ArgMap::const_iterator I = MArchMap.find(Arg);

      if (I != MArchMap.end()) {
        mattrTouched = true;
        mattr += '+';
        mattr += I->getValue();
        continue;
      }
    }

    AddPlusOrMinus(Arg, mattr);
  }

  std::string out;
  if (mattrTouched)
    out += mattr;
  if (mcpuTouched)
    out += (mattrTouched ? " " : "") + mcpu;

  return out;
}

// -mcpu values that need to be special-cased.
const char* MCpuKeysPPC[] = { "G3", "G4", "G5", "powerpc", "powerpc64"};
const char* MCpuValuesPPC[] = { "g3", "g4", "g5", "ppc", "ppc64"};
const unsigned MCpuNumKeysPPC = NUM_KEYS(MCpuKeysPPC);
const unsigned MCpuMapSize = NextHighestPowerOf2(MCpuNumKeysPPC);

/// ConvertMCpu - Convert -mcpu value from the gcc to the llc dialect.
std::string ConvertMCpu(const char* Val) {
  static ArgMap MCpuMap(MCpuMapSize);
  static bool StaticDataInitialized = false;

  if (!StaticDataInitialized) {
    FillInArgMap(MCpuMap, MCpuKeysPPC, MCpuValuesPPC, MCpuNumKeysPPC);
    StaticDataInitialized = true;
  }

  std::string ret = "-mcpu=";
  ArgMap::const_iterator I = MCpuMap.find(Val);
  if (I != MCpuMap.end()) {
    return ret + I->getValue();
  }
  return ret + Val;
}

// -mfpu values that need to be special-cased.
const char* MFpuKeysARM[] = { "vfp", "vfpv3",
                              "vfpv3-fp16", "vfpv3-d16", "vfpv3-d16-fp16",
                              "neon", "neon-fp16" };
const char* MFpuValuesARM[] = { "vfp2", "vfp3",
                                "+vfp3,+fp16", "+vfp3,+d16", "+vfp3,+d16,+fp16",
                                "+neon", "+neon,+neonfp" };
const unsigned MFpuNumKeysARM = NUM_KEYS(MFpuKeysARM);
const unsigned MFpuMapSize = NextHighestPowerOf2(MFpuNumKeysARM);

/// ConvertMFpu - Convert -mfpu value from the gcc to the llc dialect.
std::string ConvertMFpu(const char* Val) {
  static ArgMap MFpuMap(MFpuMapSize);
  static bool StaticDataInitialized = false;

  if (!StaticDataInitialized) {
    FillInArgMap(MFpuMap, MFpuKeysARM, MFpuValuesARM, MFpuNumKeysARM);
    StaticDataInitialized = true;
  }

  std::string ret = "-mattr=";
  ArgMap::const_iterator I = MFpuMap.find(Val);
  if (I != MFpuMap.end()) {
    return ret + I->getValue();
  }
  return ret + '+' + Val;
}

/// ConvertToMAttr - Convert '-mfoo' and '-mno-bar' to '-mattr=+foo,-bar'.
std::string ConvertToMAttr(const StrVec& Opts) {
  std::string out("-mattr=");
  bool firstIter = true;

  for (StrVec::const_iterator B = Opts.begin(), E = Opts.end(); B!=E; ++B) {
    const std::string& Arg = *B;

    if (firstIter)
      firstIter = false;
    else
      out += ",";

    AddPlusOrMinus(Arg, out);
  }

  return out;
}

}
