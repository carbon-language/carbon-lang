##===---- unittests/DriverTests/Makefile ----------------*- Makefile -*-===##
#
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
#
##===---------------------------------------------------------------------===##

LLD_LEVEL = ../..
TESTNAME = DriverTests
USEDLIBS = lldDriver.a \
           lldELF.a lldMachO.a lldPasses.a lldPECOFF.a lldYAML.a \
           lldReaderWriter.a lldCore.a lldNative.a \
           lldHexagonELFTarget.a lldPPCELFTarget.a lldMipsELFTarget.a \
           lldX86ELFTarget.a lldX86_64ELFTarget.a  LLVMLinker.a \
           LLVMOption.a LLVMObject.a LLVMBitReader.a LLVMCore.a LLVMSupport.a

include $(LLD_LEVEL)/unittests/Makefile
