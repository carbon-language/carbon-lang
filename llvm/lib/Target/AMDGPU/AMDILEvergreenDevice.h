//==- AMDILEvergreenDevice.h - Define Evergreen Device for AMDIL -*- C++ -*--=//
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
#ifndef _AMDILEVERGREENDEVICE_H_
#define _AMDILEVERGREENDEVICE_H_
#include "AMDILDevice.h"
#include "AMDILSubtarget.h"

namespace llvm {
  class AMDILSubtarget;
//===----------------------------------------------------------------------===//
// Evergreen generation of devices and their respective sub classes
//===----------------------------------------------------------------------===//


// The AMDILEvergreenDevice is the base device class for all of the Evergreen
// series of cards. This class contains information required to differentiate
// the Evergreen device from the generic AMDILDevice. This device represents
// that capabilities of the 'Juniper' cards, also known as the HD57XX.
class AMDILEvergreenDevice : public AMDILDevice {
public:
  AMDILEvergreenDevice(AMDILSubtarget *ST);
  virtual ~AMDILEvergreenDevice();
  virtual size_t getMaxLDSSize() const;
  virtual size_t getMaxGDSSize() const;
  virtual size_t getWavefrontSize() const;
  virtual uint32_t getGeneration() const;
  virtual uint32_t getMaxNumUAVs() const;
  virtual uint32_t getResourceID(uint32_t) const;
protected:
  virtual void setCaps();
}; // AMDILEvergreenDevice

// The AMDILCypressDevice is similiar to the AMDILEvergreenDevice, except it has
// support for double precision operations. This device is used to represent
// both the Cypress and Hemlock cards, which are commercially known as HD58XX
// and HD59XX cards.
class AMDILCypressDevice : public AMDILEvergreenDevice {
public:
  AMDILCypressDevice(AMDILSubtarget *ST);
  virtual ~AMDILCypressDevice();
private:
  virtual void setCaps();
}; // AMDILCypressDevice


// The AMDILCedarDevice is the class that represents all of the 'Cedar' based
// devices. This class differs from the base AMDILEvergreenDevice in that the
// device is a ~quarter of the 'Juniper'. These are commercially known as the
// HD54XX and HD53XX series of cards.
class AMDILCedarDevice : public AMDILEvergreenDevice {
public:
  AMDILCedarDevice(AMDILSubtarget *ST);
  virtual ~AMDILCedarDevice();
  virtual size_t getWavefrontSize() const;
private:
  virtual void setCaps();
}; // AMDILCedarDevice

// The AMDILRedwoodDevice is the class the represents all of the 'Redwood' based
// devices. This class differs from the base class, in that these devices are
// considered about half of a 'Juniper' device. These are commercially known as
// the HD55XX and HD56XX series of cards.
class AMDILRedwoodDevice : public AMDILEvergreenDevice {
public:
  AMDILRedwoodDevice(AMDILSubtarget *ST);
  virtual ~AMDILRedwoodDevice();
  virtual size_t getWavefrontSize() const;
private:
  virtual void setCaps();
}; // AMDILRedwoodDevice
  
} // namespace llvm
#endif // _AMDILEVERGREENDEVICE_H_
