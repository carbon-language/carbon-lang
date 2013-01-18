//=== MipsReginfo.h - MipsReginfo -----------------------------------------===//
//
//                    The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENCE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef MIPSREGINFO_H
#define MIPSREGINFO_H

namespace llvm {
  class MCStreamer;
  class TargetLoweringObjectFile;
  class MipsSubtarget;

  class MipsReginfo {
    void anchor();
  public:
    MipsReginfo() {}

    void emitMipsReginfoSectionCG(MCStreamer &OS,
        const TargetLoweringObjectFile &TLOF,
        const MipsSubtarget &MST) const;
  };

} // namespace llvm

#endif

