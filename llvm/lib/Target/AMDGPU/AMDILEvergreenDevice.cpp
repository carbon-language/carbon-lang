//===-- AMDILEvergreenDevice.cpp - Device Info for Evergreen --------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//==-----------------------------------------------------------------------===//
#include "AMDILEvergreenDevice.h"

using namespace llvm;

AMDILEvergreenDevice::AMDILEvergreenDevice(AMDILSubtarget *ST)
: AMDILDevice(ST) {
  setCaps();
  std::string name = ST->getDeviceName();
  if (name == "cedar") {
    mDeviceFlag = OCL_DEVICE_CEDAR;
  } else if (name == "redwood") {
    mDeviceFlag = OCL_DEVICE_REDWOOD;
  } else if (name == "cypress") {
    mDeviceFlag = OCL_DEVICE_CYPRESS;
  } else {
    mDeviceFlag = OCL_DEVICE_JUNIPER;
  }
}

AMDILEvergreenDevice::~AMDILEvergreenDevice() {
}

size_t AMDILEvergreenDevice::getMaxLDSSize() const {
  if (usesHardware(AMDILDeviceInfo::LocalMem)) {
    return MAX_LDS_SIZE_800;
  } else {
    return 0;
  }
}
size_t AMDILEvergreenDevice::getMaxGDSSize() const {
  if (usesHardware(AMDILDeviceInfo::RegionMem)) {
    return MAX_LDS_SIZE_800;
  } else {
    return 0;
  }
}
uint32_t AMDILEvergreenDevice::getMaxNumUAVs() const {
  return 12;
}

uint32_t AMDILEvergreenDevice::getResourceID(uint32_t id) const {
  switch(id) {
  default:
    assert(0 && "ID type passed in is unknown!");
    break;
  case CONSTANT_ID:
  case RAW_UAV_ID:
    if (mSTM->calVersion() >= CAL_VERSION_GLOBAL_RETURN_BUFFER) {
      return GLOBAL_RETURN_RAW_UAV_ID;
    } else {
      return DEFAULT_RAW_UAV_ID;
    }
  case GLOBAL_ID:
  case ARENA_UAV_ID:
    return DEFAULT_ARENA_UAV_ID;
  case LDS_ID:
    if (usesHardware(AMDILDeviceInfo::LocalMem)) {
      return DEFAULT_LDS_ID;
    } else {
      return DEFAULT_ARENA_UAV_ID;
    }
  case GDS_ID:
    if (usesHardware(AMDILDeviceInfo::RegionMem)) {
      return DEFAULT_GDS_ID;
    } else {
      return DEFAULT_ARENA_UAV_ID;
    }
  case SCRATCH_ID:
    if (usesHardware(AMDILDeviceInfo::PrivateMem)) {
      return DEFAULT_SCRATCH_ID;
    } else {
      return DEFAULT_ARENA_UAV_ID;
    }
  };
  return 0;
}

size_t AMDILEvergreenDevice::getWavefrontSize() const {
  return AMDILDevice::WavefrontSize;
}

uint32_t AMDILEvergreenDevice::getGeneration() const {
  return AMDILDeviceInfo::HD5XXX;
}

void AMDILEvergreenDevice::setCaps() {
  mSWBits.set(AMDILDeviceInfo::ArenaSegment);
  mHWBits.set(AMDILDeviceInfo::ArenaUAV);
  if (mSTM->calVersion() >= CAL_VERSION_SC_140) {
    mHWBits.set(AMDILDeviceInfo::HW64BitDivMod);
    mSWBits.reset(AMDILDeviceInfo::HW64BitDivMod);
  } 
  mSWBits.set(AMDILDeviceInfo::Signed24BitOps);
  if (mSTM->isOverride(AMDILDeviceInfo::ByteStores)) {
    mHWBits.set(AMDILDeviceInfo::ByteStores);
  }
  if (mSTM->isOverride(AMDILDeviceInfo::Debug)) {
    mSWBits.set(AMDILDeviceInfo::LocalMem);
    mSWBits.set(AMDILDeviceInfo::RegionMem);
  } else {
    mHWBits.set(AMDILDeviceInfo::LocalMem);
    mHWBits.set(AMDILDeviceInfo::RegionMem);
  }
  mHWBits.set(AMDILDeviceInfo::Images);
  if (mSTM->isOverride(AMDILDeviceInfo::NoAlias)) {
    mHWBits.set(AMDILDeviceInfo::NoAlias);
  }
  if (mSTM->calVersion() > CAL_VERSION_GLOBAL_RETURN_BUFFER) {
    mHWBits.set(AMDILDeviceInfo::CachedMem);
  }
  if (mSTM->isOverride(AMDILDeviceInfo::MultiUAV)) {
    mHWBits.set(AMDILDeviceInfo::MultiUAV);
  }
  if (mSTM->calVersion() > CAL_VERSION_SC_136) {
    mHWBits.set(AMDILDeviceInfo::ByteLDSOps);
    mSWBits.reset(AMDILDeviceInfo::ByteLDSOps);
    mHWBits.set(AMDILDeviceInfo::ArenaVectors);
  } else {
    mSWBits.set(AMDILDeviceInfo::ArenaVectors);
  }
  if (mSTM->calVersion() > CAL_VERSION_SC_137) {
    mHWBits.set(AMDILDeviceInfo::LongOps);
    mSWBits.reset(AMDILDeviceInfo::LongOps);
  }
  mHWBits.set(AMDILDeviceInfo::TmrReg);
}

AMDILCypressDevice::AMDILCypressDevice(AMDILSubtarget *ST)
  : AMDILEvergreenDevice(ST) {
  setCaps();
}

AMDILCypressDevice::~AMDILCypressDevice() {
}

void AMDILCypressDevice::setCaps() {
  if (mSTM->isOverride(AMDILDeviceInfo::DoubleOps)) {
    mHWBits.set(AMDILDeviceInfo::DoubleOps);
    mHWBits.set(AMDILDeviceInfo::FMA);
  }
}


AMDILCedarDevice::AMDILCedarDevice(AMDILSubtarget *ST)
  : AMDILEvergreenDevice(ST) {
  setCaps();
}

AMDILCedarDevice::~AMDILCedarDevice() {
}

void AMDILCedarDevice::setCaps() {
  mSWBits.set(AMDILDeviceInfo::FMA);
}

size_t AMDILCedarDevice::getWavefrontSize() const {
  return AMDILDevice::QuarterWavefrontSize;
}

AMDILRedwoodDevice::AMDILRedwoodDevice(AMDILSubtarget *ST)
  : AMDILEvergreenDevice(ST) {
  setCaps();
}

AMDILRedwoodDevice::~AMDILRedwoodDevice()
{
}

void AMDILRedwoodDevice::setCaps() {
  mSWBits.set(AMDILDeviceInfo::FMA);
}

size_t AMDILRedwoodDevice::getWavefrontSize() const {
  return AMDILDevice::HalfWavefrontSize;
}
