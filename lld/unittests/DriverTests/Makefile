##===---- unittests/DriverTests/Makefile ----------------*- Makefile -*-===##
#
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
#
##===-------------------------------------------------------------------===##

LLD_LEVEL = ../..
TESTNAME = DriverTests
USEDLIBS = lldDriver.a lldConfig.a \
           lldELF.a lldMachO.a lldPasses.a lldPECOFF.a \
           lldCore.a lldNative.a lldReaderWriter.a \
           lldHexagonELFTarget.a lldPPCELFTarget.a lldMipsELFTarget.a \
           lldX86ELFTarget.a lldX86_64ELFTarget.a lldYAML.a \
           lldAArch64ELFTarget.a lldARMELFTarget.a \
           LLVMObject.a LLVMMCParser.a LLVMMC.a LLVMBitReader.a \
           LLVMCore.a LLVMOption.a LLVMSupport.a
include $(LLD_LEVEL)/unittests/Makefile
