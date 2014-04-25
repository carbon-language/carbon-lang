set(LLVM_LINK_COMPONENTS support)

set(BUILDVARIABLES_SRCPATH ${CMAKE_CURRENT_SOURCE_DIR}/BuildVariables.inc.in)
set(BUILDVARIABLES_OBJPATH ${CMAKE_CURRENT_BINARY_DIR}/BuildVariables.inc)

# Compute the substitution values for various items.
get_property(LLVM_SYSTEM_LIBS_LIST TARGET LLVMSupport PROPERTY LLVM_SYSTEM_LIBS)
foreach(l ${LLVM_SYSTEM_LIBS_LIST})
  set(SYSTEM_LIBS ${SYSTEM_LIBS} "-l${l}")
endforeach()
string(REPLACE ";" " " SYSTEM_LIBS "${SYSTEM_LIBS}")

# Use configure_file to create BuildVariables.inc.
set(LLVM_SRC_ROOT ${LLVM_MAIN_SRC_DIR})
set(LLVM_OBJ_ROOT ${LLVM_BINARY_DIR})
set(LLVM_CPPFLAGS "${CMAKE_CPP_FLAGS} ${CMAKE_CPP_FLAGS_${uppercase_CMAKE_BUILD_TYPE}} ${LLVM_DEFINITIONS}")
set(LLVM_CFLAGS "${CMAKE_C_FLAGS} ${CMAKE_C_FLAGS_${uppercase_CMAKE_BUILD_TYPE}} ${LLVM_DEFINITIONS}")
set(LLVM_CXXFLAGS "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_${uppercase_CMAKE_BUILD_TYPE}} ${LLVM_DEFINITIONS}")
# Use the C++ link flags, since they should be a superset of C link flags.
set(LLVM_LDFLAGS "${CMAKE_CXX_LINK_FLAGS}")
set(LLVM_BUILDMODE ${CMAKE_BUILD_TYPE})
set(LLVM_SYSTEM_LIBS ${SYSTEM_LIBS})
string(REPLACE ";" " " LLVM_TARGETS_BUILT "${LLVM_TARGETS_TO_BUILD}")
configure_file(${BUILDVARIABLES_SRCPATH} ${BUILDVARIABLES_OBJPATH} @ONLY)

# Set build-time environment(s).
add_definitions(-DCMAKE_CFG_INTDIR="${CMAKE_CFG_INTDIR}")

# Add the llvm-config tool.
add_llvm_tool(llvm-config
  llvm-config.cpp
  )

# Add the dependency on the generation step.
add_file_dependencies(${CMAKE_CURRENT_SOURCE_DIR}/llvm-config.cpp ${BUILDVARIABLES_OBJPATH})
