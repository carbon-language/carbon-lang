//==-- AMDIL7XXDevice.h - Define 7XX Device Device for AMDIL ---*- C++ -*--===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//==-----------------------------------------------------------------------===//
//
// Interface for the subtarget data classes.
//
//===----------------------------------------------------------------------===//
// This file will define the interface that each generation needs to
// implement in order to correctly answer queries on the capabilities of the
// specific hardware.
//===----------------------------------------------------------------------===//
#ifndef _AMDIL7XXDEVICEIMPL_H_
#define _AMDIL7XXDEVICEIMPL_H_
#include "AMDILDevice.h"
#include "AMDILSubtarget.h"

namespace llvm {
class AMDILSubtarget;

//===----------------------------------------------------------------------===//
// 7XX generation of devices and their respective sub classes
//===----------------------------------------------------------------------===//

// The AMDIL7XXDevice class represents the generic 7XX device. All 7XX
// devices are derived from this class. The AMDIL7XX device will only
// support the minimal features that are required to be considered OpenCL 1.0
// compliant and nothing more.
class AMDIL7XXDevice : public AMDILDevice {
public:
  AMDIL7XXDevice(AMDILSubtarget *ST);
  virtual ~AMDIL7XXDevice();
  virtual size_t getMaxLDSSize() const;
  virtual size_t getWavefrontSize() const;
  virtual uint32_t getGeneration() const;
  virtual uint32_t getResourceID(uint32_t DeviceID) const;
  virtual uint32_t getMaxNumUAVs() const;

protected:
  virtual void setCaps();
}; // AMDIL7XXDevice

// The AMDIL770Device class represents the RV770 chip and it's
// derivative cards. The difference between this device and the base
// class is this device device adds support for double precision
// and has a larger wavefront size.
class AMDIL770Device : public AMDIL7XXDevice {
public:
  AMDIL770Device(AMDILSubtarget *ST);
  virtual ~AMDIL770Device();
  virtual size_t getWavefrontSize() const;
private:
  virtual void setCaps();
}; // AMDIL770Device

// The AMDIL710Device class derives from the 7XX base class, but this
// class is a smaller derivative, so we need to overload some of the
// functions in order to correctly specify this information.
class AMDIL710Device : public AMDIL7XXDevice {
public:
  AMDIL710Device(AMDILSubtarget *ST);
  virtual ~AMDIL710Device();
  virtual size_t getWavefrontSize() const;
}; // AMDIL710Device

} // namespace llvm
#endif // _AMDILDEVICEIMPL_H_
