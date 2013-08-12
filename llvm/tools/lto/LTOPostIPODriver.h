//===---------- LTOPostIPODriver.h - PostIPO Driver -----------------------===//
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file declare the PostIPODriver class which is the driver for 
// Post-IPO compilation phase.
//
//===----------------------------------------------------------------------===//

#ifndef LTO_POSTIPO_DRIVER_H
#define LTO_POSTIPO_DRIVER_H

#include "llvm/Target/TargetMachine.h"

namespace lto {
  class IPOPartMgr;
  class IPOFileMgr;
  class IPOFile;

  class PostIPODriver {
  public:
    typedef enum {
      PIDV_Invalid,
      PIDV_SERIAL,      // No partition
      PIDV_MultiThread, // Each partition is compiled by a thread
      PIDV_MultiProc,   // Each partition is compiled by a process
      PIDV_MakeUtil     // Partitions compilation is driven by a make-utility
    } VariantTy;

    PostIPODriver(VariantTy Var, TargetMachine *TM, IPOPartMgr &IPM,
                  IPOFileMgr &IFM, bool ToMergeObjs = false);
  
    // Return the single resulting object file. If there is no prior
    // compilation failure, this function may return NULL iff:
    //   1) Partition is enabled, and 
    //   2) Multiple partitions are generated, and
    //   3) It is not asked to merge together the objects corresponding to the
    //      the partions.
    IPOFile *getSingleObjFile() const;
  
    bool Compile(std::string &ErrMsg);

  private:
    void *DrvImpl;
    VariantTy DrvStyle;
  };
}

#endif // LTO_POSTIPO_DRIVER_H
