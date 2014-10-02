set(LLVM_NO_RTTI 1)

#add_subdirectory(arm64)
#add_subdirectory(arm)
add_subdirectory(i386)
#add_subdirectory(ppc)
add_subdirectory(x86_64)

include_directories(..)

set(generated_mach_interfaces
  ${CMAKE_CURRENT_BINARY_DIR}/mach_exc.h
  ${CMAKE_CURRENT_BINARY_DIR}/mach_excServer.c
  ${CMAKE_CURRENT_BINARY_DIR}/mach_excUser.c
  )
add_custom_command(OUTPUT ${generated_mach_interfaces}
  COMMAND mig -I${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/dbgnub-mig.defs
  DEPENDS ${dnbconfig_header}
  )

set(DEBUGSERVER_USED_LIBS
  lldbDebugserverCommon
  lldbUtility
  lldbDebugserverMacOSX_I386
  lldbDebugserverMacOSX_X86_64
  )

add_lldb_executable(debugserver
  ${dnbconfig_header}
  HasAVX.s
  CFBundle.cpp
  CFData.cpp
  CFString.cpp
  Genealogy.cpp
  MachException.cpp
  MachProcess.mm
  MachTask.mm
  MachThread.cpp
  MachThreadList.cpp
  MachVMMemory.cpp
  MachVMRegion.cpp
  ${generated_mach_interfaces}
  )

set_source_files_properties(
  HasAVX.s
  # Necessary since compilation will fail with stand-alone assembler
  PROPERTIES LANGUAGE C COMPILE_FLAGS "-x assembler-with-cpp"
  )

add_dependencies(debugserver generate_dnbconfig)
target_link_libraries(debugserver ${DEBUGSERVER_USED_LIBS})

# Sign the debugserver binary
set (CODESIGN_IDENTITY lldb_codesign)
execute_process(
  COMMAND xcrun -f codesign_allocate
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE CODESIGN_ALLOCATE
  )
add_custom_command(TARGET debugserver
  POST_BUILD
  # --entitlements option removed, as it causes errors when debugging.
  #was: COMMAND CODESIGN_ALLOCATE=${CODESIGN_ALLOCATE} codesign --entitlements ${CMAKE_CURRENT_SOURCE_DIR}/../debugserver-entitlements.plist --force --sign ${CODESIGN_IDENTITY} debugserver
  COMMAND CODESIGN_ALLOCATE=${CODESIGN_ALLOCATE} codesign --force --sign ${CODESIGN_IDENTITY} debugserver
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )

install(TARGETS debugserver
  RUNTIME DESTINATION bin
  )
