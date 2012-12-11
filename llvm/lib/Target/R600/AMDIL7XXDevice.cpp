//===-- AMDIL7XXDevice.cpp - Device Info for 7XX GPUs ---------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
// \file
//==-----------------------------------------------------------------------===//
#include "AMDIL7XXDevice.h"
#include "AMDGPUSubtarget.h"
#include "AMDILDevice.h"

using namespace llvm;

AMDGPU7XXDevice::AMDGPU7XXDevice(AMDGPUSubtarget *ST) : AMDGPUDevice(ST) {
  setCaps();
  std::string name = mSTM->getDeviceName();
  if (name == "rv710") {
    DeviceFlag = OCL_DEVICE_RV710;
  } else if (name == "rv730") {
    DeviceFlag = OCL_DEVICE_RV730;
  } else {
    DeviceFlag = OCL_DEVICE_RV770;
  }
}

AMDGPU7XXDevice::~AMDGPU7XXDevice() {
}

void AMDGPU7XXDevice::setCaps() {
  mSWBits.set(AMDGPUDeviceInfo::LocalMem);
}

size_t AMDGPU7XXDevice::getMaxLDSSize() const {
  if (usesHardware(AMDGPUDeviceInfo::LocalMem)) {
    return MAX_LDS_SIZE_700;
  }
  return 0;
}

size_t AMDGPU7XXDevice::getWavefrontSize() const {
  return AMDGPUDevice::HalfWavefrontSize;
}

uint32_t AMDGPU7XXDevice::getGeneration() const {
  return AMDGPUDeviceInfo::HD4XXX;
}

uint32_t AMDGPU7XXDevice::getResourceID(uint32_t DeviceID) const {
  switch (DeviceID) {
  default:
    assert(0 && "ID type passed in is unknown!");
    break;
  case GLOBAL_ID:
  case CONSTANT_ID:
  case RAW_UAV_ID:
  case ARENA_UAV_ID:
    break;
  case LDS_ID:
    if (usesHardware(AMDGPUDeviceInfo::LocalMem)) {
      return DEFAULT_LDS_ID;
    }
    break;
  case SCRATCH_ID:
    if (usesHardware(AMDGPUDeviceInfo::PrivateMem)) {
      return DEFAULT_SCRATCH_ID;
    }
    break;
  case GDS_ID:
    assert(0 && "GDS UAV ID is not supported on this chip");
    if (usesHardware(AMDGPUDeviceInfo::RegionMem)) {
      return DEFAULT_GDS_ID;
    }
    break;
  };

  return 0;
}

uint32_t AMDGPU7XXDevice::getMaxNumUAVs() const {
  return 1;
}

AMDGPU770Device::AMDGPU770Device(AMDGPUSubtarget *ST): AMDGPU7XXDevice(ST) {
  setCaps();
}

AMDGPU770Device::~AMDGPU770Device() {
}

void AMDGPU770Device::setCaps() {
  if (mSTM->isOverride(AMDGPUDeviceInfo::DoubleOps)) {
    mSWBits.set(AMDGPUDeviceInfo::FMA);
    mHWBits.set(AMDGPUDeviceInfo::DoubleOps);
  }
  mSWBits.set(AMDGPUDeviceInfo::BarrierDetect);
  mHWBits.reset(AMDGPUDeviceInfo::LongOps);
  mSWBits.set(AMDGPUDeviceInfo::LongOps);
  mSWBits.set(AMDGPUDeviceInfo::LocalMem);
}

size_t AMDGPU770Device::getWavefrontSize() const {
  return AMDGPUDevice::WavefrontSize;
}

AMDGPU710Device::AMDGPU710Device(AMDGPUSubtarget *ST) : AMDGPU7XXDevice(ST) {
}

AMDGPU710Device::~AMDGPU710Device() {
}

size_t AMDGPU710Device::getWavefrontSize() const {
  return AMDGPUDevice::QuarterWavefrontSize;
}
