# This data is used to establish executable/library
# dependencies.  Comes from the llvm-config script, which is built and
# installed on the bin directory for MinGW or Linux. At the end of the
# script, you'll see lines like this:

# LLVMARMAsmPrinter.o: LLVMARMCodeGen.o libLLVMAsmPrinter.a libLLVMCodeGen.a libLLVMCore.a libLLVMSupport.a libLLVMTarget.a

# This is translated to:

# set(MSVC_LIB_DEPS_LLVMARMAsmPrinter LLVMARMCodeGen LLVMAsmPrinter LLVMCodeGen LLVMCore LLVMSupport LLVMTarget)

# It is necessary to remove the `lib' prefix and the `.a'.

# This 'sed' script should do the trick:
# sed -e s'#\.a##g' -e 's#libLLVM#LLVM#g' -e 's#: ##' -e 's#\(.*\)#set(MSVC_LIB_DEPS_\1)#' ~/llvm/tools/llvm-config/LibDeps.txt
#

# TODO: do this transformations on cmake.

# It is very important that the LLVM built for extracting this data
# must contain all targets, not just X86.


set(MSVC_LIB_DEPS_LLVMARMAsmPrinter LLVMARMInfo LLVMAsmPrinter LLVMCodeGen LLVMCore LLVMSupport LLVMSystem LLVMTarget)
set(MSVC_LIB_DEPS_LLVMARMCodeGen LLVMARMInfo LLVMCodeGen LLVMCore LLVMSelectionDAG LLVMSupport LLVMSystem LLVMTarget)
set(MSVC_LIB_DEPS_LLVMARMInfo LLVMCore LLVMSupport)
set(MSVC_LIB_DEPS_LLVMAlphaAsmPrinter LLVMAlphaInfo LLVMAsmPrinter LLVMCodeGen LLVMCore LLVMSupport LLVMSystem LLVMTarget)
set(MSVC_LIB_DEPS_LLVMAlphaCodeGen LLVMAlphaInfo LLVMCodeGen LLVMCore LLVMSelectionDAG LLVMSupport LLVMSystem LLVMTarget)
set(MSVC_LIB_DEPS_LLVMAlphaInfo LLVMCore LLVMSupport)
set(MSVC_LIB_DEPS_LLVMAnalysis LLVMCore LLVMSupport LLVMSystem LLVMTarget)
set(MSVC_LIB_DEPS_LLVMArchive LLVMBitReader LLVMCore LLVMSupport LLVMSystem)
set(MSVC_LIB_DEPS_LLVMAsmParser LLVMCore LLVMSupport LLVMSystem)
set(MSVC_LIB_DEPS_LLVMAsmPrinter LLVMAnalysis LLVMCodeGen LLVMCore LLVMSupport LLVMSystem LLVMTarget)
set(MSVC_LIB_DEPS_LLVMBitReader LLVMCore LLVMSupport LLVMSystem)
set(MSVC_LIB_DEPS_LLVMBitWriter LLVMCore LLVMSupport LLVMSystem)
set(MSVC_LIB_DEPS_LLVMCBackend LLVMAnalysis LLVMCBackendInfo LLVMCodeGen LLVMCore LLVMScalarOpts LLVMSupport LLVMSystem LLVMTarget LLVMTransformUtils LLVMipa)
set(MSVC_LIB_DEPS_LLVMCBackendInfo LLVMSupport)
set(MSVC_LIB_DEPS_LLVMCellSPUAsmPrinter LLVMAsmPrinter LLVMCellSPUInfo LLVMCodeGen LLVMCore LLVMSupport LLVMSystem LLVMTarget)
set(MSVC_LIB_DEPS_LLVMCellSPUCodeGen LLVMCellSPUInfo LLVMCodeGen LLVMCore LLVMSelectionDAG LLVMSupport LLVMTarget)
set(MSVC_LIB_DEPS_LLVMCellSPUInfo LLVMSupport)
set(MSVC_LIB_DEPS_LLVMCodeGen LLVMAnalysis LLVMCore LLVMScalarOpts LLVMSupport LLVMSystem LLVMTarget LLVMTransformUtils)
set(MSVC_LIB_DEPS_LLVMCore LLVMSupport LLVMSystem)
set(MSVC_LIB_DEPS_LLVMCppBackend LLVMCore LLVMCppBackendInfo LLVMSupport LLVMSystem LLVMTarget)
set(MSVC_LIB_DEPS_LLVMCppBackendInfo LLVMSupport)
set(MSVC_LIB_DEPS_LLVMDebugger LLVMAnalysis LLVMBitReader LLVMCore LLVMSupport LLVMSystem)
set(MSVC_LIB_DEPS_LLVMExecutionEngine LLVMCore LLVMSupport LLVMSystem LLVMTarget)
set(MSVC_LIB_DEPS_LLVMInstrumentation LLVMCore LLVMScalarOpts LLVMSupport LLVMSystem LLVMTransformUtils)
set(MSVC_LIB_DEPS_LLVMInterpreter LLVMCodeGen LLVMCore LLVMExecutionEngine LLVMSupport LLVMSystem LLVMTarget)
set(MSVC_LIB_DEPS_LLVMJIT LLVMCodeGen LLVMCore LLVMExecutionEngine LLVMSupport LLVMSystem LLVMTarget)
set(MSVC_LIB_DEPS_LLVMLinker LLVMArchive LLVMBitReader LLVMCore LLVMSupport LLVMSystem)
set(MSVC_LIB_DEPS_LLVMMC LLVMSupport)
set(MSVC_LIB_DEPS_LLVMMSIL LLVMAnalysis LLVMCodeGen LLVMCore LLVMMSILInfo LLVMScalarOpts LLVMSupport LLVMSystem LLVMTarget LLVMTransformUtils LLVMipa)
set(MSVC_LIB_DEPS_LLVMMSILInfo LLVMSupport)
set(MSVC_LIB_DEPS_LLVMMSP430 LLVMAsmPrinter LLVMCodeGen LLVMCore LLVMMSP430Info LLVMSelectionDAG LLVMSupport LLVMSystem LLVMTarget)
set(MSVC_LIB_DEPS_LLVMMSP430Info LLVMSupport)
set(MSVC_LIB_DEPS_LLVMMipsAsmPrinter LLVMAsmPrinter LLVMCodeGen LLVMCore LLVMMipsCodeGen LLVMMipsInfo LLVMSupport LLVMSystem LLVMTarget)
set(MSVC_LIB_DEPS_LLVMMipsCodeGen LLVMCodeGen LLVMCore LLVMMipsInfo LLVMSelectionDAG LLVMSupport LLVMSystem LLVMTarget)
set(MSVC_LIB_DEPS_LLVMMipsInfo LLVMSupport)
set(MSVC_LIB_DEPS_LLVMPIC16 LLVMAnalysis LLVMAsmPrinter LLVMCodeGen LLVMCore LLVMPIC16Info LLVMSelectionDAG LLVMSupport LLVMSystem LLVMTarget)
set(MSVC_LIB_DEPS_LLVMPIC16Info LLVMSupport)
set(MSVC_LIB_DEPS_LLVMPowerPCAsmPrinter LLVMAsmPrinter LLVMCodeGen LLVMCore LLVMPowerPCInfo LLVMSupport LLVMSystem LLVMTarget)
set(MSVC_LIB_DEPS_LLVMPowerPCCodeGen LLVMCodeGen LLVMCore LLVMPowerPCInfo LLVMSelectionDAG LLVMSupport LLVMSystem LLVMTarget)
set(MSVC_LIB_DEPS_LLVMPowerPCInfo LLVMCore LLVMSupport)
set(MSVC_LIB_DEPS_LLVMScalarOpts LLVMAnalysis LLVMCore LLVMSupport LLVMSystem LLVMTarget LLVMTransformUtils)
set(MSVC_LIB_DEPS_LLVMSelectionDAG LLVMAnalysis LLVMAsmPrinter LLVMCodeGen LLVMCore LLVMSupport LLVMSystem LLVMTarget)
set(MSVC_LIB_DEPS_LLVMSparcAsmPrinter LLVMAsmPrinter LLVMCodeGen LLVMCore LLVMSparcInfo LLVMSupport LLVMSystem LLVMTarget)
set(MSVC_LIB_DEPS_LLVMSparcCodeGen LLVMCodeGen LLVMCore LLVMSelectionDAG LLVMSparcInfo LLVMSupport LLVMSystem LLVMTarget)
set(MSVC_LIB_DEPS_LLVMSparcInfo LLVMCore LLVMSupport)
set(MSVC_LIB_DEPS_LLVMSupport LLVMSystem)
set(MSVC_LIB_DEPS_LLVMSystem)
set(MSVC_LIB_DEPS_LLVMTarget LLVMCore LLVMSupport LLVMSystem)
set(MSVC_LIB_DEPS_LLVMTransformUtils LLVMAnalysis LLVMCore LLVMSupport LLVMSystem LLVMTarget LLVMipa)
set(MSVC_LIB_DEPS_LLVMX86AsmPrinter LLVMAsmPrinter LLVMCodeGen LLVMCore LLVMMC LLVMSupport LLVMSystem LLVMTarget LLVMX86CodeGen LLVMX86Info)
set(MSVC_LIB_DEPS_LLVMX86CodeGen LLVMCodeGen LLVMCore LLVMSelectionDAG LLVMSupport LLVMSystem LLVMTarget LLVMX86Info)
set(MSVC_LIB_DEPS_LLVMX86Info LLVMCore LLVMSupport)
set(MSVC_LIB_DEPS_LLVMXCore LLVMAsmPrinter LLVMCodeGen LLVMCore LLVMSelectionDAG LLVMSupport LLVMSystem LLVMTarget LLVMXCoreInfo)
set(MSVC_LIB_DEPS_LLVMXCoreInfo LLVMSupport)
set(MSVC_LIB_DEPS_LLVMipa LLVMAnalysis LLVMCore LLVMSupport LLVMSystem)
set(MSVC_LIB_DEPS_LLVMipo LLVMAnalysis LLVMCore LLVMSupport LLVMSystem LLVMTarget LLVMTransformUtils LLVMipa)
