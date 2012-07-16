//===-- AMDILDeviceInfo.cpp - AMDILDeviceInfo class -----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//==-----------------------------------------------------------------------===//
//
// Function that creates DeviceInfo from a device name and other information.
//
//==-----------------------------------------------------------------------===//
#include "AMDILDevices.h"
#include "AMDILSubtarget.h"

using namespace llvm;
namespace llvm {
namespace AMDILDeviceInfo {
    AMDILDevice*
getDeviceFromName(const std::string &deviceName, AMDILSubtarget *ptr, bool is64bit, bool is64on32bit)
{
    if (deviceName.c_str()[2] == '7') {
        switch (deviceName.c_str()[3]) {
            case '1':
                return new AMDIL710Device(ptr);
            case '7':
                return new AMDIL770Device(ptr);
            default:
                return new AMDIL7XXDevice(ptr);
        };
    } else if (deviceName == "cypress") {
#if DEBUG
      assert(!is64bit && "This device does not support 64bit pointers!");
      assert(!is64on32bit && "This device does not support 64bit"
          " on 32bit pointers!");
#endif
        return new AMDILCypressDevice(ptr);
    } else if (deviceName == "juniper") {
#if DEBUG
      assert(!is64bit && "This device does not support 64bit pointers!");
      assert(!is64on32bit && "This device does not support 64bit"
          " on 32bit pointers!");
#endif
        return new AMDILEvergreenDevice(ptr);
    } else if (deviceName == "redwood") {
#if DEBUG
      assert(!is64bit && "This device does not support 64bit pointers!");
      assert(!is64on32bit && "This device does not support 64bit"
          " on 32bit pointers!");
#endif
      return new AMDILRedwoodDevice(ptr);
    } else if (deviceName == "cedar") {
#if DEBUG
      assert(!is64bit && "This device does not support 64bit pointers!");
      assert(!is64on32bit && "This device does not support 64bit"
          " on 32bit pointers!");
#endif
        return new AMDILCedarDevice(ptr);
    } else if (deviceName == "barts"
      || deviceName == "turks") {
#if DEBUG
      assert(!is64bit && "This device does not support 64bit pointers!");
      assert(!is64on32bit && "This device does not support 64bit"
          " on 32bit pointers!");
#endif
        return new AMDILNIDevice(ptr);
    } else if (deviceName == "cayman") {
#if DEBUG
      assert(!is64bit && "This device does not support 64bit pointers!");
      assert(!is64on32bit && "This device does not support 64bit"
          " on 32bit pointers!");
#endif
        return new AMDILCaymanDevice(ptr);
    } else if (deviceName == "caicos") {
#if DEBUG
      assert(!is64bit && "This device does not support 64bit pointers!");
      assert(!is64on32bit && "This device does not support 64bit"
          " on 32bit pointers!");
#endif
        return new AMDILNIDevice(ptr);
    } else if (deviceName == "SI") {
        return new AMDILSIDevice(ptr);
    } else {
#if DEBUG
      assert(!is64bit && "This device does not support 64bit pointers!");
      assert(!is64on32bit && "This device does not support 64bit"
          " on 32bit pointers!");
#endif
        return new AMDIL7XXDevice(ptr);
    }
}
} // End namespace AMDILDeviceInfo
} // End namespace llvm
