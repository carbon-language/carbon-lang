//===-- AMDILNIDevice.cpp - Device Info for Northern Islands devices ------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//==-----------------------------------------------------------------------===//
#include "AMDILNIDevice.h"
#include "AMDILEvergreenDevice.h"
#include "AMDILSubtarget.h"

using namespace llvm;

AMDILNIDevice::AMDILNIDevice(AMDILSubtarget *ST)
  : AMDILEvergreenDevice(ST)
{
  std::string name = ST->getDeviceName();
  if (name == "caicos") {
    mDeviceFlag = OCL_DEVICE_CAICOS;
  } else if (name == "turks") {
    mDeviceFlag = OCL_DEVICE_TURKS;
  } else if (name == "cayman") {
    mDeviceFlag = OCL_DEVICE_CAYMAN;
  } else {
    mDeviceFlag = OCL_DEVICE_BARTS;
  }
}
AMDILNIDevice::~AMDILNIDevice()
{
}

size_t
AMDILNIDevice::getMaxLDSSize() const
{
  if (usesHardware(AMDILDeviceInfo::LocalMem)) {
    return MAX_LDS_SIZE_900;
  } else {
    return 0;
  }
}

uint32_t
AMDILNIDevice::getGeneration() const
{
  return AMDILDeviceInfo::HD6XXX;
}


AMDILCaymanDevice::AMDILCaymanDevice(AMDILSubtarget *ST)
  : AMDILNIDevice(ST)
{
  setCaps();
}

AMDILCaymanDevice::~AMDILCaymanDevice()
{
}

void
AMDILCaymanDevice::setCaps()
{
  if (mSTM->isOverride(AMDILDeviceInfo::DoubleOps)) {
    mHWBits.set(AMDILDeviceInfo::DoubleOps);
    mHWBits.set(AMDILDeviceInfo::FMA);
  }
  mHWBits.set(AMDILDeviceInfo::Signed24BitOps);
  mSWBits.reset(AMDILDeviceInfo::Signed24BitOps);
  mSWBits.set(AMDILDeviceInfo::ArenaSegment);
}

