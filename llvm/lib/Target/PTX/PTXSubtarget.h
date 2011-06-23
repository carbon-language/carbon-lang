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
    public:

      /**
       * Enumeration of Shader Models supported by the back-end.
       */
      enum PTXShaderModelEnum {
        PTX_SM_1_0, /*< Shader Model 1.0 */
        PTX_SM_1_3, /*< Shader Model 1.3 */
        PTX_SM_2_0  /*< Shader Model 2.0 */
      };

      /**
       * Enumeration of PTX versions supported by the back-end.
       *
       * Currently, PTX 2.0 is the minimum supported version.
       */
      enum PTXVersionEnum {
        PTX_VERSION_2_0,  /*< PTX Version 2.0 */
        PTX_VERSION_2_1,  /*< PTX Version 2.1 */
        PTX_VERSION_2_2,  /*< PTX Version 2.2 */
        PTX_VERSION_2_3   /*< PTX Version 2.3 */
      };

  private:

      /// Shader Model supported on the target GPU.
      PTXShaderModelEnum PTXShaderModel;

      /// PTX Language Version.
      PTXVersionEnum PTXVersion;

      // The native .f64 type is supported on the hardware.
      bool SupportsDouble;

      // Support the fused-multiply add (FMA) and multiply-add (MAD)
      // instructions
      bool SupportsFMA;

      // Use .u64 instead of .u32 for addresses.
      bool Is64Bit;

    public:

      PTXSubtarget(const std::string &TT, const std::string &FS, bool is64Bit);

      // Target architecture accessors
      std::string getTargetString() const;

      std::string getPTXVersionString() const;

      bool supportsDouble() const { return SupportsDouble; }

      bool is64Bit() const { return Is64Bit; }

      bool supportsFMA() const { return SupportsFMA; }

      bool supportsSM13() const { return PTXShaderModel >= PTX_SM_1_3; }

      bool supportsSM20() const { return PTXShaderModel >= PTX_SM_2_0; }

      bool supportsPTX21() const { return PTXVersion >= PTX_VERSION_2_1; }

      bool supportsPTX22() const { return PTXVersion >= PTX_VERSION_2_2; }

      bool supportsPTX23() const { return PTXVersion >= PTX_VERSION_2_3; }

      PTXShaderModelEnum getShaderModel() const { return PTXShaderModel; }


      std::string ParseSubtargetFeatures(const std::string &FS,
                                         const std::string &CPU);
  }; // class PTXSubtarget
} // namespace llvm

#endif // PTX_SUBTARGET_H
