//===------- AMDILSIDevice.h - Define SI Device for AMDIL -*- C++ -*------===//
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
///
/// This file will define the interface that each generation needs to
/// implement in order to correctly answer queries on the capabilities of the
/// specific hardware.
//===---------------------------------------------------------------------===//
#ifndef AMDILSIDEVICE_H
#define AMDILSIDEVICE_H
#include "AMDILEvergreenDevice.h"

namespace llvm {
class AMDGPUSubtarget;
//===---------------------------------------------------------------------===//
// SI generation of devices and their respective sub classes
//===---------------------------------------------------------------------===//

/// \brief The AMDGPUSIDevice is the base class for all Southern Island series
/// of cards.
class AMDGPUSIDevice : public AMDGPUEvergreenDevice {
public:
  AMDGPUSIDevice(AMDGPUSubtarget*);
  virtual ~AMDGPUSIDevice();
  virtual size_t getMaxLDSSize() const;
  virtual uint32_t getGeneration() const;
  virtual std::string getDataLayout() const;
};

} // namespace llvm
#endif // AMDILSIDEVICE_H
