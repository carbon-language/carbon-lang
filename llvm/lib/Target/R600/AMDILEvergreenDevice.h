//==- AMDILEvergreenDevice.h - Define Evergreen Device for AMDIL -*- C++ -*--=//
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
//===----------------------------------------------------------------------===//
#ifndef AMDILEVERGREENDEVICE_H
#define AMDILEVERGREENDEVICE_H
#include "AMDILDevice.h"
#include "AMDGPUSubtarget.h"

namespace llvm {
  class AMDGPUSubtarget;
//===----------------------------------------------------------------------===//
// Evergreen generation of devices and their respective sub classes
//===----------------------------------------------------------------------===//


/// \brief The AMDGPUEvergreenDevice is the base device class for all of the Evergreen
/// series of cards.
///
/// This class contains information required to differentiate
/// the Evergreen device from the generic AMDGPUDevice. This device represents
/// that capabilities of the 'Juniper' cards, also known as the HD57XX.
class AMDGPUEvergreenDevice : public AMDGPUDevice {
public:
  AMDGPUEvergreenDevice(AMDGPUSubtarget *ST);
  virtual ~AMDGPUEvergreenDevice();
  virtual size_t getMaxLDSSize() const;
  virtual size_t getMaxGDSSize() const;
  virtual size_t getWavefrontSize() const;
  virtual uint32_t getGeneration() const;
  virtual uint32_t getMaxNumUAVs() const;
  virtual uint32_t getResourceID(uint32_t) const;
protected:
  virtual void setCaps();
};

/// The AMDGPUCypressDevice is similiar to the AMDGPUEvergreenDevice, except it has
/// support for double precision operations. This device is used to represent
/// both the Cypress and Hemlock cards, which are commercially known as HD58XX
/// and HD59XX cards.
class AMDGPUCypressDevice : public AMDGPUEvergreenDevice {
public:
  AMDGPUCypressDevice(AMDGPUSubtarget *ST);
  virtual ~AMDGPUCypressDevice();
private:
  virtual void setCaps();
};


/// \brief The AMDGPUCedarDevice is the class that represents all of the 'Cedar' based
/// devices.
///
/// This class differs from the base AMDGPUEvergreenDevice in that the
/// device is a ~quarter of the 'Juniper'. These are commercially known as the
/// HD54XX and HD53XX series of cards.
class AMDGPUCedarDevice : public AMDGPUEvergreenDevice {
public:
  AMDGPUCedarDevice(AMDGPUSubtarget *ST);
  virtual ~AMDGPUCedarDevice();
  virtual size_t getWavefrontSize() const;
private:
  virtual void setCaps();
};

/// \brief The AMDGPURedwoodDevice is the class the represents all of the 'Redwood' based
/// devices.
///
/// This class differs from the base class, in that these devices are
/// considered about half of a 'Juniper' device. These are commercially known as
/// the HD55XX and HD56XX series of cards.
class AMDGPURedwoodDevice : public AMDGPUEvergreenDevice {
public:
  AMDGPURedwoodDevice(AMDGPUSubtarget *ST);
  virtual ~AMDGPURedwoodDevice();
  virtual size_t getWavefrontSize() const;
private:
  virtual void setCaps();
};
  
} // namespace llvm
#endif // AMDILEVERGREENDEVICE_H
