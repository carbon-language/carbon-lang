//===------- AMDILNIDevice.h - Define NI Device for AMDIL -*- C++ -*------===//
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
//===---------------------------------------------------------------------===//
#ifndef AMDILNIDEVICE_H
#define AMDILNIDEVICE_H
#include "AMDILEvergreenDevice.h"
#include "AMDGPUSubtarget.h"

namespace llvm {

class AMDGPUSubtarget;
//===---------------------------------------------------------------------===//
// NI generation of devices and their respective sub classes
//===---------------------------------------------------------------------===//

/// \brief The AMDGPUNIDevice is the base class for all Northern Island series of
/// cards.
///
/// It is very similiar to the AMDGPUEvergreenDevice, with the major
/// exception being differences in wavefront size and hardware capabilities.  The
/// NI devices are all 64 wide wavefronts and also add support for signed 24 bit
/// integer operations
class AMDGPUNIDevice : public AMDGPUEvergreenDevice {
public:
  AMDGPUNIDevice(AMDGPUSubtarget*);
  virtual ~AMDGPUNIDevice();
  virtual size_t getMaxLDSSize() const;
  virtual uint32_t getGeneration() const;
};

/// Just as the AMDGPUCypressDevice is the double capable version of the
/// AMDGPUEvergreenDevice, the AMDGPUCaymanDevice is the double capable version
/// of the AMDGPUNIDevice.  The other major difference is that the Cayman Device
/// has 4 wide ALU's, whereas the rest of the NI family is a 5 wide.
class AMDGPUCaymanDevice: public AMDGPUNIDevice {
public:
  AMDGPUCaymanDevice(AMDGPUSubtarget*);
  virtual ~AMDGPUCaymanDevice();
private:
  virtual void setCaps();
};

static const unsigned int MAX_LDS_SIZE_900 = AMDGPUDevice::MAX_LDS_SIZE_800;
} // namespace llvm
#endif // AMDILNIDEVICE_H
