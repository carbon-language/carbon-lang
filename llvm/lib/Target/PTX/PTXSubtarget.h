//====-- PTXSubtarget.h - Define Subtarget for the PTX ---------*- C++ -*--===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares the PTX specific subclass of TargetSubtargetInfo.
//
//===----------------------------------------------------------------------===//

#ifndef PTX_SUBTARGET_H
#define PTX_SUBTARGET_H

#include "llvm/Target/TargetSubtargetInfo.h"

#define GET_SUBTARGETINFO_HEADER
#include "PTXGenSubtargetInfo.inc"

namespace llvm {
class StringRef;

  class PTXSubtarget : public PTXGenSubtargetInfo {
    public:

      /**
       * Enumeration of Shader Models supported by the back-end.
       */
      enum PTXTargetEnum {
        PTX_COMPUTE_1_0, /*< Compute Compatibility 1.0 */
        PTX_COMPUTE_1_1, /*< Compute Compatibility 1.1 */
        PTX_COMPUTE_1_2, /*< Compute Compatibility 1.2 */
        PTX_COMPUTE_1_3, /*< Compute Compatibility 1.3 */
        PTX_COMPUTE_2_0, /*< Compute Compatibility 2.0 */
        PTX_LAST_COMPUTE,

        PTX_SM_1_0, /*< Shader Model 1.0 */
        PTX_SM_1_1, /*< Shader Model 1.1 */
        PTX_SM_1_2, /*< Shader Model 1.2 */
        PTX_SM_1_3, /*< Shader Model 1.3 */
        PTX_SM_2_0, /*< Shader Model 2.0 */
        PTX_SM_2_1, /*< Shader Model 2.1 */
        PTX_SM_2_2, /*< Shader Model 2.2 */
        PTX_SM_2_3, /*< Shader Model 2.3 */
        PTX_LAST_SM
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
      PTXTargetEnum PTXTarget;

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

      PTXSubtarget(const std::string &TT, const std::string &CPU,
                   const std::string &FS, bool is64Bit);

      // Target architecture accessors
      std::string getTargetString() const;

      std::string getPTXVersionString() const;

      bool supportsDouble() const { return SupportsDouble; }

      bool is64Bit() const { return Is64Bit; }

      bool supportsFMA() const { return SupportsFMA; }

      bool supportsPTX21() const { return PTXVersion >= PTX_VERSION_2_1; }

      bool supportsPTX22() const { return PTXVersion >= PTX_VERSION_2_2; }

      bool supportsPTX23() const { return PTXVersion >= PTX_VERSION_2_3; }

      bool fdivNeedsRoundingMode() const {
        return (PTXTarget >= PTX_SM_1_3 && PTXTarget < PTX_LAST_SM) ||
               (PTXTarget >= PTX_COMPUTE_1_3 && PTXTarget < PTX_LAST_COMPUTE);
      }

      bool fmadNeedsRoundingMode() const {
        return (PTXTarget >= PTX_SM_1_3 && PTXTarget < PTX_LAST_SM) ||
               (PTXTarget >= PTX_COMPUTE_1_3 && PTXTarget < PTX_LAST_COMPUTE);
      }

      bool useParamSpaceForDeviceArgs() const {
        return (PTXTarget >= PTX_SM_2_0 && PTXTarget < PTX_LAST_SM) ||
               (PTXTarget >= PTX_COMPUTE_2_0 && PTXTarget < PTX_LAST_COMPUTE);
      }

      bool callsAreHandled() const {
        return (PTXTarget >= PTX_SM_2_0 && PTXTarget < PTX_LAST_SM) ||
               (PTXTarget >= PTX_COMPUTE_2_0 && PTXTarget < PTX_LAST_COMPUTE);
      }

      bool emitPtrAttribute() const {
        return PTXVersion >= PTX_VERSION_2_2;
      }

      void ParseSubtargetFeatures(StringRef CPU, StringRef FS);
  }; // class PTXSubtarget
} // namespace llvm

#endif // PTX_SUBTARGET_H
