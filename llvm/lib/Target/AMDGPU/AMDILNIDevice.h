//===------- AMDILNIDevice.h - Define NI Device for AMDIL -*- C++ -*------===//
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
//===---------------------------------------------------------------------===//
// This file will define the interface that each generation needs to
// implement in order to correctly answer queries on the capabilities of the
// specific hardware.
//===---------------------------------------------------------------------===//
#ifndef _AMDILNIDEVICE_H_
#define _AMDILNIDEVICE_H_
#include "AMDILEvergreenDevice.h"
#include "AMDILSubtarget.h"

namespace llvm {
  class AMDILSubtarget;
//===---------------------------------------------------------------------===//
// NI generation of devices and their respective sub classes
//===---------------------------------------------------------------------===//

// The AMDILNIDevice is the base class for all Northern Island series of
// cards. It is very similiar to the AMDILEvergreenDevice, with the major
// exception being differences in wavefront size and hardware capabilities.  The
// NI devices are all 64 wide wavefronts and also add support for signed 24 bit
// integer operations

  class AMDILNIDevice : public AMDILEvergreenDevice {
    public:
      AMDILNIDevice(AMDILSubtarget*);
      virtual ~AMDILNIDevice();
      virtual size_t getMaxLDSSize() const;
      virtual uint32_t getGeneration() const;
    protected:
  }; // AMDILNIDevice

// Just as the AMDILCypressDevice is the double capable version of the
// AMDILEvergreenDevice, the AMDILCaymanDevice is the double capable version of
// the AMDILNIDevice.  The other major difference that is not as useful from
// standpoint is that the Cayman Device has 4 wide ALU's, whereas the rest of the
// NI family is a 5 wide.
     
  class AMDILCaymanDevice: public AMDILNIDevice {
    public:
      AMDILCaymanDevice(AMDILSubtarget*);
      virtual ~AMDILCaymanDevice();
    private:
      virtual void setCaps();
  }; // AMDILCaymanDevice

  static const unsigned int MAX_LDS_SIZE_900 = AMDILDevice::MAX_LDS_SIZE_800;
} // namespace llvm
#endif // _AMDILNIDEVICE_H_
