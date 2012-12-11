//===---- AMDILDevice.h - Define Device Data for AMDGPU -----*- C++ -*------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//==-----------------------------------------------------------------------===//
//
/// \file
/// \brief Interface for the subtarget data classes.
//
/// This file will define the interface that each generation needs to
/// implement in order to correctly answer queries on the capabilities of the
/// specific hardware.
//===----------------------------------------------------------------------===//
#ifndef AMDILDEVICEIMPL_H
#define AMDILDEVICEIMPL_H
#include "AMDIL.h"
#include "llvm/ADT/BitVector.h"

namespace llvm {
  class AMDGPUSubtarget;
  class MCStreamer;
//===----------------------------------------------------------------------===//
// Interface for data that is specific to a single device
//===----------------------------------------------------------------------===//
class AMDGPUDevice {
public:
  AMDGPUDevice(AMDGPUSubtarget *ST);
  virtual ~AMDGPUDevice();

  // Enum values for the various memory types.
  enum {
    RAW_UAV_ID   = 0,
    ARENA_UAV_ID = 1,
    LDS_ID       = 2,
    GDS_ID       = 3,
    SCRATCH_ID   = 4,
    CONSTANT_ID  = 5,
    GLOBAL_ID    = 6,
    MAX_IDS      = 7
  } IO_TYPE_IDS;

  /// \returns The max LDS size that the hardware supports.  Size is in
  /// bytes.
  virtual size_t getMaxLDSSize() const = 0;

  /// \returns The max GDS size that the hardware supports if the GDS is
  /// supported by the hardware.  Size is in bytes.
  virtual size_t getMaxGDSSize() const;

  /// \returns The max number of hardware constant address spaces that
  /// are supported by this device.
  virtual size_t getMaxNumCBs() const;

  /// \returns The max number of bytes a single hardware constant buffer
  /// can support.  Size is in bytes.
  virtual size_t getMaxCBSize() const;

  /// \returns The max number of bytes allowed by the hardware scratch
  /// buffer.  Size is in bytes.
  virtual size_t getMaxScratchSize() const;

  /// \brief Get the flag that corresponds to the device.
  virtual uint32_t getDeviceFlag() const;

  /// \returns The number of work-items that exist in a single hardware
  /// wavefront.
  virtual size_t getWavefrontSize() const = 0;

  /// \brief Get the generational name of this specific device.
  virtual uint32_t getGeneration() const = 0;

  /// \brief Get the stack alignment of this specific device.
  virtual uint32_t getStackAlignment() const;

  /// \brief Get the resource ID for this specific device.
  virtual uint32_t getResourceID(uint32_t DeviceID) const = 0;

  /// \brief Get the max number of UAV's for this device.
  virtual uint32_t getMaxNumUAVs() const = 0;


  // API utilizing more detailed capabilities of each family of
  // cards. If a capability is supported, then either usesHardware or
  // usesSoftware returned true.  If usesHardware returned true, then
  // usesSoftware must return false for the same capability.  Hardware
  // execution means that the feature is done natively by the hardware
  // and is not emulated by the softare.  Software execution means
  // that the feature could be done in the hardware, but there is
  // software that emulates it with possibly using the hardware for
  // support since the hardware does not fully comply with OpenCL
  // specs.

  bool isSupported(AMDGPUDeviceInfo::Caps Mode) const;
  bool usesHardware(AMDGPUDeviceInfo::Caps Mode) const;
  bool usesSoftware(AMDGPUDeviceInfo::Caps Mode) const;
  virtual std::string getDataLayout() const;
  static const unsigned int MAX_LDS_SIZE_700 = 16384;
  static const unsigned int MAX_LDS_SIZE_800 = 32768;
  static const unsigned int WavefrontSize = 64;
  static const unsigned int HalfWavefrontSize = 32;
  static const unsigned int QuarterWavefrontSize = 16;
protected:
  virtual void setCaps();
  llvm::BitVector mHWBits;
  llvm::BitVector mSWBits;
  AMDGPUSubtarget *mSTM;
  uint32_t DeviceFlag;
private:
  AMDGPUDeviceInfo::ExecutionMode
  getExecutionMode(AMDGPUDeviceInfo::Caps Caps) const;
};

} // namespace llvm
#endif // AMDILDEVICEIMPL_H
