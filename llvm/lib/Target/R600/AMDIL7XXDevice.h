//==-- AMDIL7XXDevice.h - Define 7XX Device Device for AMDIL ---*- C++ -*--===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//==-----------------------------------------------------------------------===//
/// \file
/// \brief Interface for the subtarget data classes.
///
/// This file will define the interface that each generation needs to
/// implement in order to correctly answer queries on the capabilities of the
/// specific hardware.
//===----------------------------------------------------------------------===//
#ifndef AMDIL7XXDEVICEIMPL_H
#define AMDIL7XXDEVICEIMPL_H
#include "AMDILDevice.h"

namespace llvm {
class AMDGPUSubtarget;

//===----------------------------------------------------------------------===//
// 7XX generation of devices and their respective sub classes
//===----------------------------------------------------------------------===//

/// \brief The AMDGPU7XXDevice class represents the generic 7XX device.
///
/// All 7XX devices are derived from this class. The AMDGPU7XX device will only
/// support the minimal features that are required to be considered OpenCL 1.0
/// compliant and nothing more.
class AMDGPU7XXDevice : public AMDGPUDevice {
public:
  AMDGPU7XXDevice(AMDGPUSubtarget *ST);
  virtual ~AMDGPU7XXDevice();
  virtual size_t getMaxLDSSize() const;
  virtual size_t getWavefrontSize() const;
  virtual uint32_t getGeneration() const;
  virtual uint32_t getResourceID(uint32_t DeviceID) const;
  virtual uint32_t getMaxNumUAVs() const;

protected:
  virtual void setCaps();
};

/// \brief The AMDGPU770Device class represents the RV770 chip and it's
/// derivative cards.
///
/// The difference between this device and the base class is this device device
/// adds support for double precision and has a larger wavefront size.
class AMDGPU770Device : public AMDGPU7XXDevice {
public:
  AMDGPU770Device(AMDGPUSubtarget *ST);
  virtual ~AMDGPU770Device();
  virtual size_t getWavefrontSize() const;
private:
  virtual void setCaps();
};

/// \brief The AMDGPU710Device class derives from the 7XX base class.
///
/// This class is a smaller derivative, so we need to overload some of the
/// functions in order to correctly specify this information.
class AMDGPU710Device : public AMDGPU7XXDevice {
public:
  AMDGPU710Device(AMDGPUSubtarget *ST);
  virtual ~AMDGPU710Device();
  virtual size_t getWavefrontSize() const;
};

} // namespace llvm
#endif // AMDILDEVICEIMPL_H
