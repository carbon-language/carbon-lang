#include "llvm/ADT/StringMap.h"

#include <string>
#include <vector>

namespace hooks {
typedef std::vector<std::string> StrVec;
typedef llvm::StringMap<const char*> ArgMap;

/// ConvertToMAttrImpl - Common implementation of ConvertMArchToMAttr and
/// ConvertToMAttr. The optional Args parameter contains information about how
/// to transform special-cased values (for example, '-march=armv6' must be
/// forwarded as '-mattr=+v6').
std::string ConvertToMAttrImpl(const StrVec& Opts, const ArgMap* Args = 0) {
  std::string out("-mattr=");
  bool firstIter = true;

  for (StrVec::const_iterator B = Opts.begin(), E = Opts.end(); B!=E; ++B) {
    const std::string& Arg = *B;

    if (firstIter)
      firstIter = false;
    else
      out += ",";

    // Check if the argument is a special case.
    if (Args != 0) {
      ArgMap::const_iterator I = Args->find(Arg);

      if (I != Args->end()) {
        out += '+';
        out += I->getValue();
        continue;
      }
    }

    // Convert 'no-foo' to '-foo'.
    if (Arg.find("no-") == 0 && Arg[3] != 0) {
      out += '-';
      out += Arg.c_str() + 3;
    }
    // Convert 'foo' to '+foo'.
    else {
      out += '+';
      out += Arg;
    }
  }

  return out;
}

// Values needed to be special-cased by ConvertMArchToMAttr.
const char* MArchMapKeys[] = { "armv6" };
const char* MArchMapValues[] = { "v6" };
const unsigned NumMArchMapKeys = sizeof(MArchMapKeys) / sizeof(const char*);

void InitializeMArchMap(ArgMap& Args) {
  for (unsigned i = 0; i < NumMArchMapKeys; ++i) {
    // Explicit cast to StringRef here is necessary to pick up the right
    // overload.
    Args.GetOrCreateValue(llvm::StringRef(MArchMapKeys[i]), MArchMapValues[i]);
  }
}

/// ConvertMArchToMAttr - Try to convert -march from the gcc dialect to
/// something llc can understand.
std::string ConvertMArchToMAttr(const StrVec& Opts) {
  static ArgMap MArchMap(NumMArchMapKeys);
  static bool MArchMapInitialized = false;

  if (!MArchMapInitialized) {
    InitializeMArchMap(MArchMap);
    MArchMapInitialized = true;
  }

  return ConvertToMAttrImpl(Opts, &MArchMap);
}

/// ConvertToMAttr - Convert '-mfoo' and '-mno-bar' to '-mattr=+foo,-bar'.
std::string ConvertToMAttr(const StrVec& Opts) {
  return ConvertToMAttrImpl(Opts);
}

}
