//====-- PTXSubtarget.h - Define Subtarget for the PTX ---------*- C++ -*--===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares the PTX specific subclass of TargetSubtarget.
//
//===----------------------------------------------------------------------===//

#ifndef PTX_SUBTARGET_H
#define PTX_SUBTARGET_H

#include "llvm/Target/TargetSubtarget.h"

namespace llvm {
  class PTXSubtarget : public TargetSubtarget {
    private:
      enum PTXShaderModelEnum {
        PTX_SM_1_0,
        PTX_SM_1_3,
        PTX_SM_2_0
      };

      enum PTXVersionEnum {
        PTX_VERSION_1_4,
        PTX_VERSION_2_0,
        PTX_VERSION_2_1
      };

      /// Shader Model supported on the target GPU.
      PTXShaderModelEnum PTXShaderModel;

      /// PTX Language Version.
      PTXVersionEnum PTXVersion;

      // The native .f64 type is supported on the hardware.
      bool SupportsDouble;

      // Use .u64 instead of .u32 for addresses.
      bool Use64BitAddresses;

    public:
      PTXSubtarget(const std::string &TT, const std::string &FS);

      std::string getTargetString() const;

      std::string getPTXVersionString() const;

      bool supportsDouble() const { return SupportsDouble; }

      bool use64BitAddresses() const { return Use64BitAddresses; }

      bool supportsSM13() const { return PTXShaderModel >= PTX_SM_1_3; }

      bool supportsSM20() const { return PTXShaderModel >= PTX_SM_2_0; }

      bool supportsPTX20() const { return PTXVersion >= PTX_VERSION_2_0; }

      bool supportsPTX21() const { return PTXVersion >= PTX_VERSION_2_1; }

      std::string ParseSubtargetFeatures(const std::string &FS,
                                         const std::string &CPU);
  }; // class PTXSubtarget
} // namespace llvm

#endif // PTX_SUBTARGET_H
