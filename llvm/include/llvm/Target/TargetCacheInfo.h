//===-- llvm/Target/TargetCacheInfo.h ---------------------------*- C++ -*-===//
//
//  Describes properties of the target cache architecture.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TARGET_TARGETCACHEINFO_H
#define LLVM_TARGET_TARGETCACHEINFO_H

#include "Support/DataTypes.h"
#include <assert.h>

class TargetMachine;

struct TargetCacheInfo {
  const TargetMachine &target;
  TargetCacheInfo(const TargetCacheInfo&); // DO NOT IMPLEMENT
  void operator=(const TargetCacheInfo&);  // DO NOT IMPLEMENT
protected:
  unsigned int           numLevels;
  std::vector<unsigned short> cacheLineSizes;
  std::vector<unsigned int>   cacheSizes;
  std::vector<unsigned short> cacheAssoc;
  
public:
  TargetCacheInfo(const TargetMachine& tgt) : target(tgt) {
    Initialize();
  }
  virtual ~TargetCacheInfo() {}
  
  // Default parameters are:
  //    NumLevels    = 2
  //    L1: LineSize 16, Cache Size 32KB, Direct-mapped (assoc = 1)
  //    L2: LineSize 32, Cache Size 1 MB, 4-way associative
  // NOTE: Cache levels are numbered from 1 as above, not from 0.
  // 
  virtual  void     Initialize          (); // subclass to override defaults
  
  unsigned int      getNumCacheLevels   () const {
    return numLevels;
  }
  unsigned short    getCacheLineSize    (unsigned level)  const {
    assert(level <= cacheLineSizes.size() && "Invalid cache level");
    return cacheLineSizes[level-1];
  }
  unsigned int      getCacheSize        (unsigned level)  const {
    assert(level <= cacheSizes.size() && "Invalid cache level");
    return cacheSizes[level-1];
  }
  unsigned short    getCacheAssoc       (unsigned level)  const {
    assert(level <= cacheAssoc.size() && "Invalid cache level");
    return cacheAssoc[level];
  }
};

#endif
