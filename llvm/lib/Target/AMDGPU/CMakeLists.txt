set(LLVM_TARGET_DEFINITIONS AMDGPU.td)

tablegen(LLVM AMDGPUGenRegisterInfo.inc -gen-register-info)
tablegen(LLVM AMDGPUGenInstrInfo.inc -gen-instr-info)
tablegen(LLVM AMDGPUGenDAGISel.inc -gen-dag-isel)
tablegen(LLVM AMDGPUGenCallingConv.inc -gen-callingconv)
tablegen(LLVM AMDGPUGenSubtargetInfo.inc -gen-subtarget)
tablegen(LLVM AMDGPUGenIntrinsics.inc -gen-tgt-intrinsic)
tablegen(LLVM AMDGPUGenCodeEmitter.inc -gen-emitter)
tablegen(LLVM AMDGPUGenDFAPacketizer.inc -gen-dfa-packetizer)
add_public_tablegen_target(AMDGPUCommonTableGen)

add_llvm_target(AMDGPUCodeGen
  AMDIL7XXDevice.cpp
  AMDILCFGStructurizer.cpp
  AMDILDevice.cpp
  AMDILDeviceInfo.cpp
  AMDILEvergreenDevice.cpp
  AMDILFrameLowering.cpp
  AMDILInstrInfo.cpp
  AMDILIntrinsicInfo.cpp
  AMDILISelDAGToDAG.cpp
  AMDILISelLowering.cpp
  AMDILNIDevice.cpp
  AMDILPeepholeOptimizer.cpp
  AMDILRegisterInfo.cpp
  AMDILSIDevice.cpp
  AMDILSubtarget.cpp
  AMDGPUTargetMachine.cpp
  AMDGPUISelLowering.cpp
  AMDGPUConvertToISA.cpp
  AMDGPUInstrInfo.cpp
  AMDGPURegisterInfo.cpp
  AMDGPUUtil.cpp
  R600CodeEmitter.cpp
  R600InstrInfo.cpp
  R600ISelLowering.cpp
  R600KernelParameters.cpp
  R600MachineFunctionInfo.cpp
  R600RegisterInfo.cpp
  SIAssignInterpRegs.cpp
  SICodeEmitter.cpp
  SIInstrInfo.cpp
  SIISelLowering.cpp
  SIMachineFunctionInfo.cpp
  SIRegisterInfo.cpp
  )

add_dependencies(LLVMAMDGPUCodeGen intrinsics_gen)

add_subdirectory(TargetInfo)
add_subdirectory(MCTargetDesc)
