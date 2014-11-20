set(LLVM_TARGET_DEFINITIONS Hexagon.td)

tablegen(LLVM HexagonGenAsmWriter.inc -gen-asm-writer)
tablegen(LLVM HexagonGenCallingConv.inc -gen-callingconv)
tablegen(LLVM HexagonGenDAGISel.inc -gen-dag-isel)
tablegen(LLVM HexagonGenDFAPacketizer.inc -gen-dfa-packetizer)
tablegen(LLVM HexagonGenDisassemblerTables.inc -gen-disassembler)
tablegen(LLVM HexagonGenInstrInfo.inc -gen-instr-info)
tablegen(LLVM HexagonGenMCCodeEmitter.inc -gen-emitter)
tablegen(LLVM HexagonGenRegisterInfo.inc -gen-register-info)
tablegen(LLVM HexagonGenSubtargetInfo.inc -gen-subtarget)
add_public_tablegen_target(HexagonCommonTableGen)

add_llvm_target(HexagonCodeGen
  HexagonAsmPrinter.cpp
  HexagonCallingConvLower.cpp
  HexagonCFGOptimizer.cpp
  HexagonCopyToCombine.cpp
  HexagonExpandPredSpillCode.cpp
  HexagonFixupHwLoops.cpp
  HexagonFrameLowering.cpp
  HexagonHardwareLoops.cpp
  HexagonInstrInfo.cpp
  HexagonISelDAGToDAG.cpp
  HexagonISelLowering.cpp
  HexagonMachineFunctionInfo.cpp
  HexagonMachineScheduler.cpp
  HexagonMCInstLower.cpp
  HexagonNewValueJump.cpp
  HexagonPeephole.cpp
  HexagonRegisterInfo.cpp
  HexagonRemoveSZExtArgs.cpp
  HexagonSelectionDAGInfo.cpp
  HexagonSplitConst32AndConst64.cpp
  HexagonSplitTFRCondSets.cpp
  HexagonSubtarget.cpp
  HexagonTargetMachine.cpp
  HexagonTargetObjectFile.cpp
  HexagonVLIWPacketizer.cpp
)

add_subdirectory(TargetInfo)
add_subdirectory(MCTargetDesc)
add_subdirectory(Disassembler)
