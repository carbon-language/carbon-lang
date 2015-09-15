//===--- TargetTuple.cpp - Target tuple class -----------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/TargetTuple.h"
#include "llvm/Support/TargetParser.h"

using namespace llvm;

// FIXME: These should be removed as soon as the Triple member is replaced.
static TargetTuple::ArchType
convertTripleArchToTupleArch(Triple::ArchType Arch) {
  switch (Arch) {
  case Triple::UnknownArch:
    return TargetTuple::UnknownArch;
  case Triple::arm:
    return TargetTuple::arm;
  case Triple::armeb:
    return TargetTuple::armeb;
  case Triple::aarch64:
    return TargetTuple::aarch64;
  case Triple::aarch64_be:
    return TargetTuple::aarch64_be;
  case Triple::bpfel:
    return TargetTuple::bpfel;
  case Triple::bpfeb:
    return TargetTuple::bpfeb;
  case Triple::hexagon:
    return TargetTuple::hexagon;
  case Triple::mips:
    return TargetTuple::mips;
  case Triple::mipsel:
    return TargetTuple::mipsel;
  case Triple::mips64:
    return TargetTuple::mips64;
  case Triple::mips64el:
    return TargetTuple::mips64el;
  case Triple::msp430:
    return TargetTuple::msp430;
  case Triple::ppc:
    return TargetTuple::ppc;
  case Triple::ppc64:
    return TargetTuple::ppc64;
  case Triple::ppc64le:
    return TargetTuple::ppc64le;
  case Triple::r600:
    return TargetTuple::r600;
  case Triple::amdgcn:
    return TargetTuple::amdgcn;
  case Triple::sparc:
    return TargetTuple::sparc;
  case Triple::sparcv9:
    return TargetTuple::sparcv9;
  case Triple::sparcel:
    return TargetTuple::sparcel;
  case Triple::systemz:
    return TargetTuple::systemz;
  case Triple::tce:
    return TargetTuple::tce;
  case Triple::thumb:
    return TargetTuple::thumb;
  case Triple::thumbeb:
    return TargetTuple::thumbeb;
  case Triple::x86:
    return TargetTuple::x86;
  case Triple::x86_64:
    return TargetTuple::x86_64;
  case Triple::xcore:
    return TargetTuple::xcore;
  case Triple::nvptx:
    return TargetTuple::nvptx;
  case Triple::nvptx64:
    return TargetTuple::nvptx64;
  case Triple::le32:
    return TargetTuple::le32;
  case Triple::le64:
    return TargetTuple::le64;
  case Triple::amdil:
    return TargetTuple::amdil;
  case Triple::amdil64:
    return TargetTuple::amdil64;
  case Triple::hsail:
    return TargetTuple::hsail;
  case Triple::hsail64:
    return TargetTuple::hsail64;
  case Triple::spir:
    return TargetTuple::spir;
  case Triple::spir64:
    return TargetTuple::spir64;
  case Triple::kalimba:
    return TargetTuple::kalimba;
  case Triple::shave:
    return TargetTuple::shave;
  case Triple::wasm32:
    return TargetTuple::wasm32;
  case Triple::wasm64:
    return TargetTuple::wasm64;
  }
  llvm_unreachable("Unmapped architecture.");
}

// FIXME: These should be removed as soon as the Triple member is replaced.
static TargetTuple::SubArchType
convertTripleSubArchToTupleSubArch(Triple::SubArchType SubArch) {
  switch (SubArch) {
  case Triple::NoSubArch:
    return TargetTuple::NoSubArch;
  case Triple::ARMSubArch_v8_1a:
    return TargetTuple::ARMSubArch_v8_1a;
  case Triple::ARMSubArch_v8:
    return TargetTuple::ARMSubArch_v8;
  case Triple::ARMSubArch_v7:
    return TargetTuple::ARMSubArch_v7;
  case Triple::ARMSubArch_v7em:
    return TargetTuple::ARMSubArch_v7em;
  case Triple::ARMSubArch_v7m:
    return TargetTuple::ARMSubArch_v7m;
  case Triple::ARMSubArch_v7s:
    return TargetTuple::ARMSubArch_v7s;
  case Triple::ARMSubArch_v6:
    return TargetTuple::ARMSubArch_v6;
  case Triple::ARMSubArch_v6m:
    return TargetTuple::ARMSubArch_v6m;
  case Triple::ARMSubArch_v6k:
    return TargetTuple::ARMSubArch_v6k;
  case Triple::ARMSubArch_v6t2:
    return TargetTuple::ARMSubArch_v6t2;
  case Triple::ARMSubArch_v5:
    return TargetTuple::ARMSubArch_v5;
  case Triple::ARMSubArch_v5te:
    return TargetTuple::ARMSubArch_v5te;
  case Triple::ARMSubArch_v4t:
    return TargetTuple::ARMSubArch_v4t;
  case Triple::KalimbaSubArch_v3:
    return TargetTuple::KalimbaSubArch_v3;
  case Triple::KalimbaSubArch_v4:
    return TargetTuple::KalimbaSubArch_v4;
  case Triple::KalimbaSubArch_v5:
    return TargetTuple::KalimbaSubArch_v5;
  }
  llvm_unreachable("Unmapped subarchitecture.");
}

// FIXME: These should be removed as soon as the Triple member is replaced.
static TargetTuple::VendorType
convertTripleVendorToTupleVendor(Triple::VendorType Vendor) {
  switch (Vendor) {
  case Triple::UnknownVendor:
    return TargetTuple::UnknownVendor;
  case Triple::Apple:
    return TargetTuple::Apple;
  case Triple::PC:
    return TargetTuple::PC;
  case Triple::SCEI:
    return TargetTuple::SCEI;
  case Triple::BGP:
    return TargetTuple::BGP;
  case Triple::BGQ:
    return TargetTuple::BGQ;
  case Triple::Freescale:
    return TargetTuple::Freescale;
  case Triple::IBM:
    return TargetTuple::IBM;
  case Triple::ImaginationTechnologies:
    return TargetTuple::ImaginationTechnologies;
  case Triple::MipsTechnologies:
    return TargetTuple::MipsTechnologies;
  case Triple::NVIDIA:
    return TargetTuple::NVIDIA;
  case Triple::CSR:
    return TargetTuple::CSR;
  case Triple::Myriad:
    return TargetTuple::Myriad;
  }
  llvm_unreachable("Unmapped vendor.");
}

// FIXME: These should be removed as soon as the Triple member is replaced.
static TargetTuple::OSType convertTripleOSToTupleOS(Triple::OSType OS) {
  switch (OS) {
  case Triple::UnknownOS:
    return TargetTuple::UnknownOS;
  case Triple::CloudABI:
    return TargetTuple::CloudABI;
  case Triple::Darwin:
    return TargetTuple::Darwin;
  case Triple::DragonFly:
    return TargetTuple::DragonFly;
  case Triple::FreeBSD:
    return TargetTuple::FreeBSD;
  case Triple::IOS:
    return TargetTuple::IOS;
  case Triple::KFreeBSD:
    return TargetTuple::KFreeBSD;
  case Triple::Linux:
    return TargetTuple::Linux;
  case Triple::Lv2:
    return TargetTuple::Lv2;
  case Triple::MacOSX:
    return TargetTuple::MacOSX;
  case Triple::NetBSD:
    return TargetTuple::NetBSD;
  case Triple::OpenBSD:
    return TargetTuple::OpenBSD;
  case Triple::Solaris:
    return TargetTuple::Solaris;
  case Triple::Win32:
    return TargetTuple::Win32;
  case Triple::Haiku:
    return TargetTuple::Haiku;
  case Triple::Minix:
    return TargetTuple::Minix;
  case Triple::RTEMS:
    return TargetTuple::RTEMS;
  case Triple::NaCl:
    return TargetTuple::NaCl;
  case Triple::CNK:
    return TargetTuple::CNK;
  case Triple::Bitrig:
    return TargetTuple::Bitrig;
  case Triple::AIX:
    return TargetTuple::AIX;
  case Triple::CUDA:
    return TargetTuple::CUDA;
  case Triple::NVCL:
    return TargetTuple::NVCL;
  case Triple::AMDHSA:
    return TargetTuple::AMDHSA;
  case Triple::PS4:
    return TargetTuple::PS4;
  }
  llvm_unreachable("Unmapped OS.");
}

// FIXME: These should be removed as soon as the Triple member is replaced.
static TargetTuple::EnvironmentType
convertTripleEnvToTupleEnv(Triple::EnvironmentType Env) {
  switch (Env) {
  case Triple::UnknownEnvironment:
    return TargetTuple::UnknownEnvironment;
  case Triple::GNU:
    return TargetTuple::GNU;
  case Triple::GNUEABI:
    return TargetTuple::GNUEABI;
  case Triple::GNUEABIHF:
    return TargetTuple::GNUEABIHF;
  case Triple::GNUX32:
    return TargetTuple::GNUX32;
  case Triple::CODE16:
    return TargetTuple::CODE16;
  case Triple::EABI:
    return TargetTuple::EABI;
  case Triple::EABIHF:
    return TargetTuple::EABIHF;
  case Triple::Android:
    return TargetTuple::Android;
  case Triple::MSVC:
    return TargetTuple::MSVC;
  case Triple::Itanium:
    return TargetTuple::Itanium;
  case Triple::Cygnus:
    return TargetTuple::Cygnus;
  case Triple::AMDOpenCL:
    return TargetTuple::AMDOpenCL;
  case Triple::CoreCLR:
    return TargetTuple::CoreCLR;
  }
  llvm_unreachable("Unmapped Environment.");
}

// FIXME: These should be removed as soon as the Triple member is replaced.
static TargetTuple::ObjectFormatType
convertTripleObjFmtToTupleObjFmt(Triple::ObjectFormatType ObjFmt) {
  switch (ObjFmt) {
  case Triple::UnknownObjectFormat:
    return TargetTuple::UnknownObjectFormat;
  case Triple::COFF:
    return TargetTuple::COFF;
  case Triple::ELF:
    return TargetTuple::ELF;
  case Triple::MachO:
    return TargetTuple::MachO;
  }
  llvm_unreachable("Unmapped Object Format.");
}

TargetTuple::ArchType TargetTuple::getArch() const {
  return convertTripleArchToTupleArch(GnuTT.getArch());
}

TargetTuple::SubArchType TargetTuple::getSubArch() const {
  return convertTripleSubArchToTupleSubArch(GnuTT.getSubArch());
}

TargetTuple::VendorType TargetTuple::getVendor() const {
  return convertTripleVendorToTupleVendor(GnuTT.getVendor());
}

TargetTuple::OSType TargetTuple::getOS() const {
  return convertTripleOSToTupleOS(GnuTT.getOS());
}

TargetTuple::EnvironmentType TargetTuple::getEnvironment() const {
  return convertTripleEnvToTupleEnv(GnuTT.getEnvironment());
}

TargetTuple::ObjectFormatType TargetTuple::getObjectFormat() const {
  return convertTripleObjFmtToTupleObjFmt(GnuTT.getObjectFormat());
}
