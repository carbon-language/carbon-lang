LEVEL  = ..
DIRS   = Programs
include Makefile.tests

#
# Make QMTest the default for testing features and regressions
#
all:: qmtest

#
# New QMTest functionality:
#	The test suite is being transitioned over to QMTest.  Eventually, it
#	will use QMTest by default.
#

# QMTest option specifying the location of the QMTest database.
QMDB= -D $(LLVM_SRC_ROOT)/test/QMTestDB

#
# This is configuration information used by the test suite.  In QM Test, it's
# called a 'context.'
#
CONTEXT= -c srcroot=$(LLVM_SRC_ROOT) \
         -c buildroot=$(LLVM_OBJ_ROOT) \
         -c buildtype=$(CONFIGURATION) \
         -c tmpdir=$(LLVM_OBJ_ROOT)/test/tmp \
         -c coresize=0 \
         -c cc=$(CC) \
         -c cxx=$(CXX) \
         -c "llvmgcc=$(LLVMGCC)" \
         -c make=$(MAKE)

#
# Location of the QMTest program.
#
QMTEST= qmtest $(QMDB)


#
# Execute the tests
#
qmtest:: $(LLVM_OBJ_ROOT)/test/tmp register
	$(QMTEST) run -O $(LLVM_SRC_ROOT)/test/QMTestDB/expectations.qmr $(CONTEXT)

%.t:: $(LLVM_OBJ_ROOT)/test/tmp register
	$(QMTEST) run -O $(LLVM_SRC_ROOT)/test/QMTestDB/expectations.qmr $(CONTEXT) $*

#
# Create the temporary directory used by the test suite.
#
$(LLVM_OBJ_ROOT)/test/tmp::
	${MKDIR} $(LLVM_OBJ_ROOT)/test/tmp

#
# Register the python code with QMTest
#
register:: $(LLVM_SRC_ROOT)/test/QMTestDB/QMTest/llvm.pyo

$(LLVM_SRC_ROOT)/test/QMTestDB/QMTest/llvm.pyo:  $(LLVM_SRC_ROOT)/test/QMTestDB/QMTest/llvm.py
	$(QMTEST) register test llvm.TestAsmDisasm
	$(QMTEST) register test llvm.AssembleTest
	$(QMTEST) register test llvm.ConvertToCTest
	$(QMTEST) register test llvm.LLToCTest
	$(QMTEST) register test llvm.MachineCodeTest
	$(QMTEST) register test llvm.AssemblyCodeTest
	$(QMTEST) register test llvm.TestOptimizer
	$(QMTEST) register test llvm.LLITest
	$(QMTEST) register test llvm.TestRunner
	$(QMTEST) register test llvm.VerifierTest
	$(QMTEST) register test llvm.AnalyzeTest
	$(QMTEST) register test llvm.CTest
	$(QMTEST) register resource llvm.BytecodeResource

