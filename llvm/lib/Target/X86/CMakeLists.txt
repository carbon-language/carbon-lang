macro(x86tgen ofn)
  add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${ofn}
    COMMAND tblgen ${ARGN} -I ${CMAKE_CURRENT_SOURCE_DIR} -I ${CMAKE_SOURCE_DIR}/lib/Target -I ${llvm_include_path} ${CMAKE_CURRENT_SOURCE_DIR}/X86.td -o ${ofn}
    DEPENDS tblgen ${CMAKE_CURRENT_SOURCE_DIR}/X86.td
    COMMENT "Building ${ofn}..."
    )
endmacro(x86tgen)

x86tgen(X86GenRegisterInfo.h.inc -gen-register-desc-header)
x86tgen(X86GenRegisterNames.inc -gen-register-enums)
x86tgen(X86GenRegisterInfo.inc -gen-register-desc)
x86tgen(X86GenInstrNames.inc -gen-instr-enums)
x86tgen(X86GenInstrInfo.inc -gen-instr-desc)
x86tgen(X86GenAsmWriter.inc -gen-asm-writer)
x86tgen(X86GenAsmWriter1.inc -gen-asm-writer -asmwriternum=1)
x86tgen(X86GenDAGISel.inc -gen-dag-isel)
x86tgen(X86GenFastISel.inc -gen-fast-isel)
x86tgen(X86GenCallingConv.inc -gen-callingconv)
x86tgen(X86GenSubtarget.inc -gen-subtarget)

add_custom_target(X86Table_gen echo Tablegenning
  DEPENDS
  ${llvm_builded_incs_dir}/Intrinsics.gen
  ${CMAKE_CURRENT_BINARY_DIR}/X86GenRegisterInfo.h.inc
  ${CMAKE_CURRENT_BINARY_DIR}/X86GenRegisterNames.inc
  ${CMAKE_CURRENT_BINARY_DIR}/X86GenRegisterInfo.inc
  ${CMAKE_CURRENT_BINARY_DIR}/X86GenInstrNames.inc
  ${CMAKE_CURRENT_BINARY_DIR}/X86GenInstrInfo.inc
  ${CMAKE_CURRENT_BINARY_DIR}/X86GenAsmWriter.inc
  ${CMAKE_CURRENT_BINARY_DIR}/X86GenAsmWriter1.inc
  ${CMAKE_CURRENT_BINARY_DIR}/X86GenDAGISel.inc
  ${CMAKE_CURRENT_BINARY_DIR}/X86GenFastISel.inc
  ${CMAKE_CURRENT_BINARY_DIR}/X86GenCallingConv.inc
  ${CMAKE_CURRENT_BINARY_DIR}/X86GenSubtarget.inc
  )

include_directories(BEFORE ${CMAKE_CURRENT_BINARY_DIR})

add_partially_linked_object(LLVMX86CodeGen
  X86CodeEmitter.cpp
  X86ELFWriterInfo.cpp
  X86FloatingPoint.cpp
  X86ISelDAGToDAG.cpp
  X86ISelLowering.cpp
  X86InstrInfo.cpp
  X86JITInfo.cpp
  X86RegisterInfo.cpp
  X86Subtarget.cpp
  X86TargetAsmInfo.cpp
  X86TargetMachine.cpp
  X86FastISel.cpp
  )

add_dependencies(LLVMX86CodeGen
  X86Table_gen
)
