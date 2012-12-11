//===-- AMDILEvergreenDevice.cpp - Device Info for Evergreen --------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
/// \file
//==-----------------------------------------------------------------------===//
#include "AMDILEvergreenDevice.h"

using namespace llvm;

AMDGPUEvergreenDevice::AMDGPUEvergreenDevice(AMDGPUSubtarget *ST)
: AMDGPUDevice(ST) {
  setCaps();
  std::string name = ST->getDeviceName();
  if (name == "cedar") {
    DeviceFlag = OCL_DEVICE_CEDAR;
  } else if (name == "redwood") {
    DeviceFlag = OCL_DEVICE_REDWOOD;
  } else if (name == "cypress") {
    DeviceFlag = OCL_DEVICE_CYPRESS;
  } else {
    DeviceFlag = OCL_DEVICE_JUNIPER;
  }
}

AMDGPUEvergreenDevice::~AMDGPUEvergreenDevice() {
}

size_t AMDGPUEvergreenDevice::getMaxLDSSize() const {
  if (usesHardware(AMDGPUDeviceInfo::LocalMem)) {
    return MAX_LDS_SIZE_800;
  } else {
    return 0;
  }
}
size_t AMDGPUEvergreenDevice::getMaxGDSSize() const {
  if (usesHardware(AMDGPUDeviceInfo::RegionMem)) {
    return MAX_LDS_SIZE_800;
  } else {
    return 0;
  }
}
uint32_t AMDGPUEvergreenDevice::getMaxNumUAVs() const {
  return 12;
}

uint32_t AMDGPUEvergreenDevice::getResourceID(uint32_t id) const {
  switch(id) {
  default:
    assert(0 && "ID type passed in is unknown!");
    break;
  case CONSTANT_ID:
  case RAW_UAV_ID:
    return GLOBAL_RETURN_RAW_UAV_ID;
  case GLOBAL_ID:
  case ARENA_UAV_ID:
    return DEFAULT_ARENA_UAV_ID;
  case LDS_ID:
    if (usesHardware(AMDGPUDeviceInfo::LocalMem)) {
      return DEFAULT_LDS_ID;
    } else {
      return DEFAULT_ARENA_UAV_ID;
    }
  case GDS_ID:
    if (usesHardware(AMDGPUDeviceInfo::RegionMem)) {
      return DEFAULT_GDS_ID;
    } else {
      return DEFAULT_ARENA_UAV_ID;
    }
  case SCRATCH_ID:
    if (usesHardware(AMDGPUDeviceInfo::PrivateMem)) {
      return DEFAULT_SCRATCH_ID;
    } else {
      return DEFAULT_ARENA_UAV_ID;
    }
  };
  return 0;
}

size_t AMDGPUEvergreenDevice::getWavefrontSize() const {
  return AMDGPUDevice::WavefrontSize;
}

uint32_t AMDGPUEvergreenDevice::getGeneration() const {
  return AMDGPUDeviceInfo::HD5XXX;
}

void AMDGPUEvergreenDevice::setCaps() {
  mSWBits.set(AMDGPUDeviceInfo::ArenaSegment);
  mHWBits.set(AMDGPUDeviceInfo::ArenaUAV);
  mHWBits.set(AMDGPUDeviceInfo::HW64BitDivMod);
  mSWBits.reset(AMDGPUDeviceInfo::HW64BitDivMod);
  mSWBits.set(AMDGPUDeviceInfo::Signed24BitOps);
  if (mSTM->isOverride(AMDGPUDeviceInfo::ByteStores)) {
    mHWBits.set(AMDGPUDeviceInfo::ByteStores);
  }
  if (mSTM->isOverride(AMDGPUDeviceInfo::Debug)) {
    mSWBits.set(AMDGPUDeviceInfo::LocalMem);
    mSWBits.set(AMDGPUDeviceInfo::RegionMem);
  } else {
    mHWBits.set(AMDGPUDeviceInfo::LocalMem);
    mHWBits.set(AMDGPUDeviceInfo::RegionMem);
  }
  mHWBits.set(AMDGPUDeviceInfo::Images);
  if (mSTM->isOverride(AMDGPUDeviceInfo::NoAlias)) {
    mHWBits.set(AMDGPUDeviceInfo::NoAlias);
  }
  mHWBits.set(AMDGPUDeviceInfo::CachedMem);
  if (mSTM->isOverride(AMDGPUDeviceInfo::MultiUAV)) {
    mHWBits.set(AMDGPUDeviceInfo::MultiUAV);
  }
  mHWBits.set(AMDGPUDeviceInfo::ByteLDSOps);
  mSWBits.reset(AMDGPUDeviceInfo::ByteLDSOps);
  mHWBits.set(AMDGPUDeviceInfo::ArenaVectors);
  mHWBits.set(AMDGPUDeviceInfo::LongOps);
  mSWBits.reset(AMDGPUDeviceInfo::LongOps);
  mHWBits.set(AMDGPUDeviceInfo::TmrReg);
}

AMDGPUCypressDevice::AMDGPUCypressDevice(AMDGPUSubtarget *ST)
  : AMDGPUEvergreenDevice(ST) {
  setCaps();
}

AMDGPUCypressDevice::~AMDGPUCypressDevice() {
}

void AMDGPUCypressDevice::setCaps() {
  if (mSTM->isOverride(AMDGPUDeviceInfo::DoubleOps)) {
    mHWBits.set(AMDGPUDeviceInfo::DoubleOps);
    mHWBits.set(AMDGPUDeviceInfo::FMA);
  }
}


AMDGPUCedarDevice::AMDGPUCedarDevice(AMDGPUSubtarget *ST)
  : AMDGPUEvergreenDevice(ST) {
  setCaps();
}

AMDGPUCedarDevice::~AMDGPUCedarDevice() {
}

void AMDGPUCedarDevice::setCaps() {
  mSWBits.set(AMDGPUDeviceInfo::FMA);
}

size_t AMDGPUCedarDevice::getWavefrontSize() const {
  return AMDGPUDevice::QuarterWavefrontSize;
}

AMDGPURedwoodDevice::AMDGPURedwoodDevice(AMDGPUSubtarget *ST)
  : AMDGPUEvergreenDevice(ST) {
  setCaps();
}

AMDGPURedwoodDevice::~AMDGPURedwoodDevice() {
}

void AMDGPURedwoodDevice::setCaps() {
  mSWBits.set(AMDGPUDeviceInfo::FMA);
}

size_t AMDGPURedwoodDevice::getWavefrontSize() const {
  return AMDGPUDevice::HalfWavefrontSize;
}
