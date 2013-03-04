//===-- AMDILDevice.cpp - Base class for AMDIL Devices --------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
/// \file
//==-----------------------------------------------------------------------===//
#include "AMDILDevice.h"
#include "AMDGPUSubtarget.h"

using namespace llvm;
// Default implementation for all of the classes.
AMDGPUDevice::AMDGPUDevice(AMDGPUSubtarget *ST) : mSTM(ST) {
  mHWBits.resize(AMDGPUDeviceInfo::MaxNumberCapabilities);
  mSWBits.resize(AMDGPUDeviceInfo::MaxNumberCapabilities);
  setCaps();
  DeviceFlag = OCL_DEVICE_ALL;
}

AMDGPUDevice::~AMDGPUDevice() {
    mHWBits.clear();
    mSWBits.clear();
}

size_t AMDGPUDevice::getMaxGDSSize() const {
  return 0;
}

uint32_t 
AMDGPUDevice::getDeviceFlag() const {
  return DeviceFlag;
}

size_t AMDGPUDevice::getMaxNumCBs() const {
  if (usesHardware(AMDGPUDeviceInfo::ConstantMem)) {
    return HW_MAX_NUM_CB;
  }

  return 0;
}

size_t AMDGPUDevice::getMaxCBSize() const {
  if (usesHardware(AMDGPUDeviceInfo::ConstantMem)) {
    return MAX_CB_SIZE;
  }

  return 0;
}

size_t AMDGPUDevice::getMaxScratchSize() const {
  return 65536;
}

uint32_t AMDGPUDevice::getStackAlignment() const {
  return 16;
}

void AMDGPUDevice::setCaps() {
  mSWBits.set(AMDGPUDeviceInfo::HalfOps);
  mSWBits.set(AMDGPUDeviceInfo::ByteOps);
  mSWBits.set(AMDGPUDeviceInfo::ShortOps);
  mSWBits.set(AMDGPUDeviceInfo::HW64BitDivMod);
  if (mSTM->isOverride(AMDGPUDeviceInfo::NoInline)) {
    mSWBits.set(AMDGPUDeviceInfo::NoInline);
  }
  if (mSTM->isOverride(AMDGPUDeviceInfo::MacroDB)) {
    mSWBits.set(AMDGPUDeviceInfo::MacroDB);
  }
  if (mSTM->isOverride(AMDGPUDeviceInfo::Debug)) {
    mSWBits.set(AMDGPUDeviceInfo::ConstantMem);
  } else {
    mHWBits.set(AMDGPUDeviceInfo::ConstantMem);
  }
  if (mSTM->isOverride(AMDGPUDeviceInfo::Debug)) {
    mSWBits.set(AMDGPUDeviceInfo::PrivateMem);
  } else {
    mHWBits.set(AMDGPUDeviceInfo::PrivateMem);
  }
  if (mSTM->isOverride(AMDGPUDeviceInfo::BarrierDetect)) {
    mSWBits.set(AMDGPUDeviceInfo::BarrierDetect);
  }
  mSWBits.set(AMDGPUDeviceInfo::ByteLDSOps);
  mSWBits.set(AMDGPUDeviceInfo::LongOps);
}

AMDGPUDeviceInfo::ExecutionMode
AMDGPUDevice::getExecutionMode(AMDGPUDeviceInfo::Caps Caps) const {
  if (mHWBits[Caps]) {
    assert(!mSWBits[Caps] && "Cannot set both SW and HW caps");
    return AMDGPUDeviceInfo::Hardware;
  }

  if (mSWBits[Caps]) {
    assert(!mHWBits[Caps] && "Cannot set both SW and HW caps");
    return AMDGPUDeviceInfo::Software;
  }

  return AMDGPUDeviceInfo::Unsupported;

}

bool AMDGPUDevice::isSupported(AMDGPUDeviceInfo::Caps Mode) const {
  return getExecutionMode(Mode) != AMDGPUDeviceInfo::Unsupported;
}

bool AMDGPUDevice::usesHardware(AMDGPUDeviceInfo::Caps Mode) const {
  return getExecutionMode(Mode) == AMDGPUDeviceInfo::Hardware;
}

bool AMDGPUDevice::usesSoftware(AMDGPUDeviceInfo::Caps Mode) const {
  return getExecutionMode(Mode) == AMDGPUDeviceInfo::Software;
}

std::string
AMDGPUDevice::getDataLayout() const {
  std::string DataLayout = std::string(
   "e"
   "-p:32:32:32"
   "-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32"
   "-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128"
   "-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-v2048:2048:2048"
   "-n32:64"
  );

  if (usesHardware(AMDGPUDeviceInfo::DoubleOps)) {
    DataLayout.append("-f64:64:64");
  }

  return DataLayout;
}
