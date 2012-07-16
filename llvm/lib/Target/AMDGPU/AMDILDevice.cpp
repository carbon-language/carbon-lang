//===-- AMDILDevice.cpp - Base class for AMDIL Devices --------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//==-----------------------------------------------------------------------===//
#include "AMDILDevice.h"
#include "AMDILSubtarget.h"

using namespace llvm;
// Default implementation for all of the classes.
AMDILDevice::AMDILDevice(AMDILSubtarget *ST) : mSTM(ST)
{
  mHWBits.resize(AMDILDeviceInfo::MaxNumberCapabilities);
  mSWBits.resize(AMDILDeviceInfo::MaxNumberCapabilities);
  setCaps();
  mDeviceFlag = OCL_DEVICE_ALL;
}

AMDILDevice::~AMDILDevice()
{
    mHWBits.clear();
    mSWBits.clear();
}

size_t AMDILDevice::getMaxGDSSize() const
{
  return 0;
}

uint32_t 
AMDILDevice::getDeviceFlag() const
{
  return mDeviceFlag;
}

size_t AMDILDevice::getMaxNumCBs() const
{
  if (usesHardware(AMDILDeviceInfo::ConstantMem)) {
    return HW_MAX_NUM_CB;
  }

  return 0;
}

size_t AMDILDevice::getMaxCBSize() const
{
  if (usesHardware(AMDILDeviceInfo::ConstantMem)) {
    return MAX_CB_SIZE;
  }

  return 0;
}

size_t AMDILDevice::getMaxScratchSize() const
{
  return 65536;
}

uint32_t AMDILDevice::getStackAlignment() const
{
  return 16;
}

void AMDILDevice::setCaps()
{
  mSWBits.set(AMDILDeviceInfo::HalfOps);
  mSWBits.set(AMDILDeviceInfo::ByteOps);
  mSWBits.set(AMDILDeviceInfo::ShortOps);
  mSWBits.set(AMDILDeviceInfo::HW64BitDivMod);
  if (mSTM->isOverride(AMDILDeviceInfo::NoInline)) {
    mSWBits.set(AMDILDeviceInfo::NoInline);
  }
  if (mSTM->isOverride(AMDILDeviceInfo::MacroDB)) {
    mSWBits.set(AMDILDeviceInfo::MacroDB);
  }
  if (mSTM->isOverride(AMDILDeviceInfo::Debug)) {
    mSWBits.set(AMDILDeviceInfo::ConstantMem);
  } else {
    mHWBits.set(AMDILDeviceInfo::ConstantMem);
  }
  if (mSTM->isOverride(AMDILDeviceInfo::Debug)) {
    mSWBits.set(AMDILDeviceInfo::PrivateMem);
  } else {
    mHWBits.set(AMDILDeviceInfo::PrivateMem);
  }
  if (mSTM->isOverride(AMDILDeviceInfo::BarrierDetect)) {
    mSWBits.set(AMDILDeviceInfo::BarrierDetect);
  }
  mSWBits.set(AMDILDeviceInfo::ByteLDSOps);
  mSWBits.set(AMDILDeviceInfo::LongOps);
}

AMDILDeviceInfo::ExecutionMode
AMDILDevice::getExecutionMode(AMDILDeviceInfo::Caps Caps) const
{
  if (mHWBits[Caps]) {
    assert(!mSWBits[Caps] && "Cannot set both SW and HW caps");
    return AMDILDeviceInfo::Hardware;
  }

  if (mSWBits[Caps]) {
    assert(!mHWBits[Caps] && "Cannot set both SW and HW caps");
    return AMDILDeviceInfo::Software;
  }

  return AMDILDeviceInfo::Unsupported;

}

bool AMDILDevice::isSupported(AMDILDeviceInfo::Caps Mode) const
{
  return getExecutionMode(Mode) != AMDILDeviceInfo::Unsupported;
}

bool AMDILDevice::usesHardware(AMDILDeviceInfo::Caps Mode) const
{
  return getExecutionMode(Mode) == AMDILDeviceInfo::Hardware;
}

bool AMDILDevice::usesSoftware(AMDILDeviceInfo::Caps Mode) const
{
  return getExecutionMode(Mode) == AMDILDeviceInfo::Software;
}

std::string
AMDILDevice::getDataLayout() const
{
    return std::string("e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16"
      "-i32:32:32-i64:64:64-f32:32:32-f64:64:64-f80:32:32"
      "-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64"
      "-v96:128:128-v128:128:128-v192:256:256-v256:256:256"
      "-v512:512:512-v1024:1024:1024-v2048:2048:2048"
      "-n8:16:32:64");
}
